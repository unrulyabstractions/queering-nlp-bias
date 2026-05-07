"""Integrate new judgment questions into an existing experiment in place.

Re-scores every trajectory of every arm (and greedy.json entries, if
present) on the supplied `--categorical` / `--graded` questions, then
patches:

    * `scoring.json`           — appends questions to `scoring_items`
                                 and per-result `method_scores` /
                                 `method_raw`.
    * `scoring_cfg.json`       — appends questions.
    * `greedy.json`            — extends each arm's `structure_scores`.

After the data is patched, runs `estimate_normativity` and
`visualize_estimation` so all downstream artifacts (estimation.json,
plots, KDE, method comparison) are regenerated as if the experiment had
originally been run with the new judgments.

Usage:
    uv run python scripts/integrate_new_judgments.py \\
        out/nurse/nurse/scoring.json \\
        --categorical "Is the nurse a man?" "Is the nurse a woman?"

    uv run python scripts/integrate_new_judgments.py \\
        out/nurse/nurse/scoring.json \\
        --graded "How nurturing is the nurse?" \\
        --judge anthropic/sonnet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_section

from judgment_scoring_helpers import (  # noqa: E402
    ArmRescore,
    build_scorer,
    list_arms,
    load_greedy_alongside,
    load_scoring,
    score_all_arms,
)


# ─── Patchers ────────────────────────────────────────────────────────────────


def patch_scoring_json(
    scoring_path: Path,
    rescores: dict[str, ArmRescore],
    *,
    categorical: list[str],
    graded: list[str],
) -> None:
    """Append new questions and per-result scores to scoring.json in place."""
    with open(scoring_path, encoding="utf-8") as f:
        data = json.load(f)

    items = data.setdefault("scoring_items", {})
    if categorical:
        items.setdefault("categorical_judgements", []).extend(categorical)
    if graded:
        items.setdefault("graded_judgements", []).extend(graded)

    # Build (arm, traj_idx) -> (cat_scores, cat_raw, grade_scores, grade_raw).
    n_cat = len(categorical)
    n_grade = len(graded)
    by_key: dict[tuple[str, int], tuple[list[float], list[str], list[float], list[str]]] = {}
    for arm, rs in rescores.items():
        for i, traj_idx in enumerate(rs.sample_traj_indices):
            scores = rs.sample_scores[i] if i < len(rs.sample_scores) else []
            raw = rs.sample_raw[i] if i < len(rs.sample_raw) else []
            cat_s = scores[:n_cat]
            cat_r = raw[:n_cat]
            gr_s = scores[n_cat : n_cat + n_grade]
            gr_r = raw[n_cat : n_cat + n_grade]
            by_key[(arm, int(traj_idx))] = (cat_s, cat_r, gr_s, gr_r)

    for r in data["results"]:
        key = (r.get("arm"), int(r["traj_idx"]))
        if key not in by_key:
            continue
        cat_s, cat_r, gr_s, gr_r = by_key[key]
        ms = r.setdefault("method_scores", {})
        mr = r.setdefault("method_raw", {})
        if categorical:
            ms.setdefault("categorical", []).extend(int(s) if s in (0, 1) else float(s) for s in cat_s)
            mr.setdefault("categorical", []).extend(cat_r)
        if graded:
            ms.setdefault("graded", []).extend(float(s) for s in gr_s)
            mr.setdefault("graded", []).extend(gr_r)

    with open(scoring_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def patch_scoring_cfg(
    scoring_path: Path,
    *,
    categorical: list[str],
    graded: list[str],
) -> None:
    """Append questions to `scoring_cfg.json` next to the scoring file."""
    cfg_path = scoring_path.parent / "scoring_cfg.json"
    if not cfg_path.exists():
        return
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    if categorical:
        cfg.setdefault("categorical_judgements", []).extend(categorical)
    if graded:
        cfg.setdefault("graded_judgements", []).extend(graded)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def patch_greedy_json(
    scoring_path: Path,
    rescores: dict[str, ArmRescore],
    *,
    n_old_cat: int,
    n_old_grade: int,
    n_new_cat: int,
    n_new_grade: int,
) -> None:
    """Insert new question scores into each arm's `structure_scores` so the
    flat list stays in canonical order: [all-categorical..., all-graded...].
    """
    greedy_path, greedy = load_greedy_alongside(scoring_path)
    if greedy is None or greedy_path is None:
        return
    for arm, rs in rescores.items():
        if not rs.greedy_scores:
            continue
        entry = greedy.get_arm(arm)
        if entry is None:
            continue

        old = list(entry.structure_scores)
        old_cat = old[:n_old_cat]
        old_grade = old[n_old_cat : n_old_cat + n_old_grade]

        new_scores = list(rs.greedy_scores)
        new_cat = new_scores[:n_new_cat]
        new_grade = new_scores[n_new_cat : n_new_cat + n_new_grade]

        entry.structure_scores = old_cat + new_cat + old_grade + new_grade
    greedy.save(greedy_path)


# ─── Pipeline reruns ─────────────────────────────────────────────────────────


def rerun_estimation_and_viz(scoring_path: Path) -> None:
    """Invoke estimation + visualization scripts so all artifacts refresh."""
    repo_root = Path(__file__).parent.parent

    log_section("Re-running estimation")
    subprocess.run(
        ["uv", "run", "python", "scripts/estimate_normativity.py", str(scoring_path)],
        cwd=repo_root,
        check=True,
    )

    log_section("Re-running visualization")
    estimation_path = scoring_path.parent / "estimation.json"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/visualize_estimation.py",
            str(estimation_path),
        ],
        cwd=repo_root,
        check=True,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("scoring_path", type=Path)
    p.add_argument("--categorical", nargs="*", default=[])
    p.add_argument("--graded", nargs="*", default=[])
    p.add_argument("--judge", default="anthropic/sonnet")
    p.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Patch data files only; skip estimation + viz reruns.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.categorical and not args.graded:
        raise SystemExit("Provide at least one --categorical or --graded question.")

    log_section("Integrate new judgments")
    log(f"Scoring file: {args.scoring_path}")
    log(f"Categorical:  {args.categorical}")
    log(f"Graded:       {args.graded}")
    log(f"Judge:        {args.judge}")

    scoring_data = load_scoring(args.scoring_path)
    _, greedy = load_greedy_alongside(args.scoring_path)
    if greedy is None:
        log("(no greedy.json found; greedy entries will not be patched)")

    # Snapshot original counts so we can keep greedy.json's flat
    # `structure_scores` in canonical [all-categorical..., all-graded...] order.
    items = scoring_data.get("scoring_items", {})
    n_old_cat = len(items.get("categorical_judgements", []))
    n_old_grade = len(items.get("graded_judgements", []))

    scorer = build_scorer(
        judge_model=args.judge,
        categorical=args.categorical,
        graded=args.graded,
    )

    arms = list_arms(scoring_data)
    log(f"\nProcessing arms: {arms}")
    rescores = score_all_arms(scorer, scoring_data, greedy, log_fn=log)

    log_section("Patching data files")
    patch_scoring_json(
        args.scoring_path,
        rescores,
        categorical=args.categorical,
        graded=args.graded,
    )
    log(f"  scoring.json     ← {args.scoring_path}")

    patch_scoring_cfg(
        args.scoring_path,
        categorical=args.categorical,
        graded=args.graded,
    )
    log(f"  scoring_cfg.json ← {args.scoring_path.parent / 'scoring_cfg.json'}")

    patch_greedy_json(
        args.scoring_path,
        rescores,
        n_old_cat=n_old_cat,
        n_old_grade=n_old_grade,
        n_new_cat=len(args.categorical),
        n_new_grade=len(args.graded),
    )
    log("  greedy.json      ← updated structure_scores per arm")

    if args.skip_pipeline:
        log("\n--skip-pipeline set; not re-running estimation or visualization")
        return

    rerun_estimation_and_viz(args.scoring_path)


if __name__ == "__main__":
    main()
