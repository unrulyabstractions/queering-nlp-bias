"""Score one generation output with multiple judges in parallel,
fanning out across arms.

For each (judge, arm) pair we spawn an isolated subprocess running
`score_trajectories.py --arm <arm> --skip-greedy`. Once every per-arm
job for a judge finishes, this script merges the per-arm sidecars into
a single `scoring.json` for that judge and runs greedy scoring once
at the end.

This re-shards a long-tailed run (where one arm dominates wall time)
into N×M shorter independent jobs, exploiting the fact that the slow
arm only needs to run once per judge instead of blocking the rest.

Usage:
    uv run python scripts/score_judges_parallel.py \\
        --gen out/nurse_sub/generation.json \\
        --judges trials/scoring/opus.json trials/scoring/gpt5.json \\
                 trials/scoring/gemini.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_section
from src.generation.greedy_output import GreedyOutput
from src.scoring import GenerationOutputData, ScoringConfig, ScoringOutput
from src.scoring.scorer import Scorer


REPO_ROOT = Path(__file__).parent.parent
SCORE_TRAJECTORIES = REPO_ROOT / "scripts" / "score_trajectories.py"


@dataclass
class JobSpec:
    """One (judge, arm) scoring job."""

    judge_cfg: Path
    arm: str
    log_path: Path

    @property
    def label(self) -> str:
        return f"{self.judge_cfg.stem}/{self.arm}"


# ─── Subprocess execution ────────────────────────────────────────────────


def _run_one_job(spec: JobSpec, gen_path: Path) -> tuple[JobSpec, int, float]:
    """Run `score_trajectories.py --arm <arm> --skip-greedy` and return rc."""
    started = time.time()
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec.log_path, "w") as fh:
        proc = subprocess.run(
            [
                "uv", "run", "python", str(SCORE_TRAJECTORIES),
                str(spec.judge_cfg),
                str(gen_path),
                "--arm", spec.arm,
                "--skip-greedy",
            ],
            cwd=REPO_ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return spec, proc.returncode, time.time() - started


# ─── Per-judge merge ─────────────────────────────────────────────────────


def _merge_arm_files(
    scoring_dir: Path,
    arm_names: list[str],
    judge_cfg: Path,
) -> Path:
    """Concatenate per-arm sidecar scoring.json files into one scoring.json.

    Each `_arm_<arm>/scoring.json` is a complete ScoringOutput restricted
    to that arm. We pick one as the template (for metadata + scoring_items)
    and union their `results` lists in arm order.
    """
    parts: list[dict] = []
    for arm in arm_names:
        p = scoring_dir / f"_arm_{arm}" / "scoring.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing per-arm sidecar: {p}")
        with open(p, encoding="utf-8") as f:
            parts.append(json.load(f))

    merged_results: list[dict] = []
    for part in parts:
        merged_results.extend(part.get("results", []))
    merged_results.sort(key=lambda r: r.get("traj_idx", 0))

    base = parts[0]
    merged = {
        "arm_names": base["arm_names"],
        "arm_texts": base["arm_texts"],
        "metadata": base["metadata"],
        "scoring_items": base["scoring_items"],
        "results": merged_results,
    }

    out_path = scoring_dir / "scoring.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    # Also drop a scoring_cfg.json for consistency with single-arm runs.
    cfg_dst = scoring_dir / "scoring_cfg.json"
    if not cfg_dst.exists():
        shutil.copy(judge_cfg, cfg_dst)

    return out_path


def _score_greedy_for_judge(
    judge_cfg: Path,
    scoring_dir: Path,
    gen_path: Path,
) -> Path | None:
    """Score `<gen_dir>/greedy.json` once with this judge; write sidecar."""
    greedy_src = GreedyOutput.compute_path(gen_path)
    if not greedy_src.exists():
        return None

    config = ScoringConfig.load(judge_cfg)
    greedy_output = GreedyOutput.load(greedy_src)
    scorer = Scorer(config)
    for entry in greedy_output.arms:
        entry.structure_scores = scorer.score(entry.text)
    out_path = scoring_dir / "greedy.json"
    greedy_output.save(out_path)
    return out_path


def _cleanup_arm_sidecars(scoring_dir: Path, arm_names: list[str]) -> None:
    """Remove `_arm_<arm>/` subdirs once their data is merged."""
    for arm in arm_names:
        d = scoring_dir / f"_arm_{arm}"
        if d.exists():
            shutil.rmtree(d)


# ─── Orchestration ───────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen", type=Path, required=True, help="Path to generation.json")
    p.add_argument(
        "--judges",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to scoring config JSONs (one per judge).",
    )
    p.add_argument(
        "--max-parallel",
        type=int,
        default=15,
        help="Max concurrent subprocesses across all (judge, arm) jobs.",
    )
    p.add_argument(
        "--keep-arm-sidecars",
        action="store_true",
        help="Don't delete `_arm_<arm>/` subdirs after merging.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_section("Per-arm parallel scoring")
    log(f"Generation: {args.gen}")
    log(f"Judges:     {[j.stem for j in args.judges]}")

    gen_data = GenerationOutputData.load(args.gen)
    arm_names = gen_data.arm_names
    log(f"Arms:       {arm_names}")
    log(f"Trajs:      {len(gen_data.trajectories)} ({len(gen_data.trajectories)//max(len(arm_names),1)}/arm)")

    # One JobSpec per (judge, arm). Per-judge log dir keeps things readable.
    log_root = Path("/tmp") / "score_judges_parallel"
    log_root.mkdir(parents=True, exist_ok=True)

    jobs: list[JobSpec] = []
    for judge_cfg in args.judges:
        for arm in arm_names:
            log_path = log_root / f"{judge_cfg.stem}__{arm}.log"
            jobs.append(JobSpec(judge_cfg=judge_cfg, arm=arm, log_path=log_path))

    log(f"\nDispatching {len(jobs)} subprocesses (max parallel: {args.max_parallel})")

    failures: list[tuple[JobSpec, int]] = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futures = {ex.submit(_run_one_job, j, args.gen): j for j in jobs}
        for fut in as_completed(futures):
            spec, rc, elapsed = fut.result()
            status = "OK" if rc == 0 else f"FAILED ({rc})"
            log(f"  [{spec.label}] {status} in {elapsed:.0f}s ({spec.log_path})")
            if rc != 0:
                failures.append((spec, rc))

    if failures:
        log_section("Aborting: some jobs failed")
        for spec, rc in failures:
            log(f"  {spec.label}: rc={rc}; see {spec.log_path}")
        sys.exit(1)

    # Merge per-arm sidecars per judge, then score greedy.json once.
    log_section("Merging per-arm sidecars + scoring greedy paths")
    for judge_cfg in args.judges:
        scoring_dir = (
            ScoringOutput.compute_output_path(args.gen, judge_cfg).parent
        )
        merged = _merge_arm_files(scoring_dir, arm_names, judge_cfg)
        log(f"  {judge_cfg.stem}: merged → {merged}")

        greedy_out = _score_greedy_for_judge(judge_cfg, scoring_dir, args.gen)
        if greedy_out:
            log(f"  {judge_cfg.stem}: greedy → {greedy_out}")

        if not args.keep_arm_sidecars:
            _cleanup_arm_sidecars(scoring_dir, arm_names)

    log_section("Done")


if __name__ == "__main__":
    main()
