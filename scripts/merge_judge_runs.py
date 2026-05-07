"""Merge per-judge scoring runs into a single ensemble scoring directory.

Reads scoring.json from each of `--judges` (subfolders of the same
generation tree) and writes:

* `scoring.json`        — one entry per trajectory, with each method's
                          per-question scores averaged across judges
                          (treats 0/1 categorical labels as floats so the
                          ensemble produces a soft probability).
* `scoring_cfg.json`    — questions copied from the first judge.
* `greedy.json`         — averaged per-arm greedy structure_scores when
                          per-judge sidecars are present.

The result lives at `<gen_dir>/<scoring_name>/` and can be fed to
`estimate_normativity.py` + `visualize_estimation.py` like any other
scoring run.

Usage:
    uv run python scripts/merge_judge_runs.py \\
        --gen-dir out/nurse \\
        --judges opus gpt5 gemini \\
        --name all
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_section


def load_scoring(p: Path) -> dict:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def average_scores(values: list[list[float]]) -> list[float]:
    """Per-position average across N judges' score vectors. Skips empty rows."""
    if not values:
        return []
    width = max(len(v) for v in values)
    out: list[float] = []
    for i in range(width):
        column = [float(v[i]) for v in values if i < len(v) and v[i] is not None]
        out.append(sum(column) / len(column) if column else 0.0)
    return out


def merge_method_scores(per_judge: list[dict]) -> tuple[dict, dict]:
    """Average each method's per-question scores across judges.

    Returns (averaged_method_scores, merged_method_raw_per_judge).
    """
    methods: set[str] = set()
    for d in per_judge:
        methods.update(d.keys())

    averaged: dict[str, list[float]] = {}
    raw_by_judge: dict[str, list[dict]] = {}
    for method in methods:
        per_method = [d.get(method, []) for d in per_judge]
        averaged[method] = average_scores(per_method)
        # Keep raw per-judge for traceability
        raw_by_judge[method] = per_method
    return averaged, raw_by_judge


def merge_scoring(
    gen_dir: Path,
    judges: list[str],
    name: str,
    judge_name: str = "ensemble",
) -> Path:
    log_section(f"Merging {len(judges)} judges → {gen_dir / name}")
    runs = []
    for j in judges:
        path = gen_dir / j / "scoring.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing scoring file: {path}")
        runs.append((j, path, load_scoring(path)))
        log(f"  loaded {j}: {path}")

    base = runs[0][2]
    out_dir = gen_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index each judge's results by traj_idx
    judge_index: list[tuple[str, dict[int, dict]]] = []
    for j, _path, data in runs:
        idx = {int(r["traj_idx"]): r for r in data["results"]}
        judge_index.append((j, idx))

    merged_results = []
    skipped = 0
    for r in base["results"]:
        traj_idx = int(r["traj_idx"])
        per_method_lists = []
        per_method_raw_by_judge: dict[str, dict[str, list]] = defaultdict(dict)
        all_present = True
        for j, idx in judge_index:
            entry = idx.get(traj_idx)
            if entry is None:
                all_present = False
                break
            per_method_lists.append(entry.get("method_scores", {}))
            for m, vals in entry.get("method_raw", {}).items():
                per_method_raw_by_judge[m][j] = vals
        if not all_present:
            skipped += 1
            continue

        avg, _ = merge_method_scores(per_method_lists)
        merged_results.append({
            **{k: v for k, v in r.items() if k not in ("method_scores", "method_raw")},
            "method_scores": avg,
            "method_raw": {
                m: {"_judges": list(per_method_raw_by_judge[m].keys()),
                    **per_method_raw_by_judge[m]}
                for m in per_method_raw_by_judge
            },
        })

    if skipped:
        log(f"  skipped {skipped} trajectories (not in every judge run)")

    merged = {
        "arm_names": base["arm_names"],
        "arm_texts": base["arm_texts"],
        "metadata": {
            **base["metadata"],
            "judge_model": f"{judge_name}({'+'.join(judges)})",
            "scoring_file": str(out_dir / "scoring.json"),
        },
        "scoring_items": base["scoring_items"],
        "results": merged_results,
    }
    out_path = out_dir / "scoring.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    log(f"  wrote {out_path}")

    # Mirror scoring_cfg.json so downstream viz can read it.
    src_cfg = gen_dir / judges[0] / "scoring_cfg.json"
    if src_cfg.exists():
        dst_cfg = out_dir / "scoring_cfg.json"
        with open(src_cfg, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["model"] = merged["metadata"]["judge_model"]
        with open(dst_cfg, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        log(f"  wrote {dst_cfg}")

    # Merge per-judge greedy.json sidecars if all are present.
    greedy_paths = [gen_dir / j / "greedy.json" for j in judges]
    if all(p.exists() for p in greedy_paths):
        greedy_runs = [json.load(open(p, encoding="utf-8")) for p in greedy_paths]
        merged_arms = []
        # Build by arm name across judges
        arm_names = [a["name"] for a in greedy_runs[0]["arms"]]
        for arm_name in arm_names:
            scores_per_judge = []
            base_entry = None
            for run in greedy_runs:
                entry = next((a for a in run["arms"] if a["name"] == arm_name), None)
                if entry is None:
                    break
                scores_per_judge.append(entry.get("structure_scores", []))
                if base_entry is None:
                    base_entry = dict(entry)
            if base_entry and scores_per_judge:
                base_entry["structure_scores"] = average_scores(scores_per_judge)
                merged_arms.append(base_entry)
        out_greedy = {
            "version": greedy_runs[0].get("version", "1.0"),
            "metadata": {
                **greedy_runs[0].get("metadata", {}),
                "judge_model": merged["metadata"]["judge_model"],
            },
            "arms": merged_arms,
        }
        greedy_out = out_dir / "greedy.json"
        with open(greedy_out, "w", encoding="utf-8") as f:
            json.dump(out_greedy, f, indent=2, ensure_ascii=False)
        log(f"  wrote {greedy_out} (averaged across {len(judges)} judges)")
    else:
        missing = [str(p) for p in greedy_paths if not p.exists()]
        log(f"  skipping greedy merge (missing: {missing})")

    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gen-dir", type=Path, required=True)
    p.add_argument("--judges", nargs="+", required=True)
    p.add_argument("--name", default="all", help="Subfolder name for the merged run.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_scoring(args.gen_dir, args.judges, args.name)


if __name__ == "__main__":
    main()
