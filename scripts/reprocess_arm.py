"""Re-score one or more arms with new judgment questions and report cores.

Read-only: prints per-arm cores under each method (uniform/avg, greedy,
mode, median). Use `integrate_new_judgments.py` to write the results back
into `scoring.json`, `scoring_cfg.json`, `greedy.json` and re-run
estimation + visualization.

Usage:
    uv run python scripts/reprocess_arm.py out/nurse/nurse/scoring.json \\
        --arm trunk \\
        --categorical "Is the nurse a man?" "Is the nurse a woman?"

    uv run python scripts/reprocess_arm.py out/nurse/nurse/scoring.json \\
        --arm all \\
        --graded "How nurturing is the nurse?"

    uv run python scripts/reprocess_arm.py out/nurse/nurse/scoring.json \\
        --arm all \\
        --categorical Q1 Q2 \\
        --graded Q3 Q4 \\
        --judge anthropic/sonnet \\
        --out cores.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_section

from judgment_scoring_helpers import (  # noqa: E402  (post sys.path tweak)
    build_scorer,
    compute_cores,
    list_arms,
    load_greedy_alongside,
    load_scoring,
    score_all_arms,
    score_arm,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scoring_path", type=Path)
    p.add_argument("--arm", default="all", help="Arm name or 'all' (default).")
    p.add_argument("--categorical", nargs="*", default=[], help="Yes/No questions.")
    p.add_argument("--graded", nargs="*", default=[], help="0..1 graded questions.")
    p.add_argument(
        "--judge",
        default="anthropic/sonnet",
        help="Judge model (default: anthropic/sonnet).",
    )
    p.add_argument("--out", type=Path, help="Optional path to dump per-arm cores as JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.categorical and not args.graded:
        raise SystemExit("Provide at least one --categorical or --graded question.")

    log_section("Reprocess arm with new judgments")
    log(f"Scoring file: {args.scoring_path}")
    log(f"Categorical: {args.categorical}")
    log(f"Graded:      {args.graded}")
    log(f"Judge:       {args.judge}")

    scoring_data = load_scoring(args.scoring_path)
    _, greedy = load_greedy_alongside(args.scoring_path)
    if greedy is None:
        log("(no greedy.json found; greedy core falls back to average)")

    scorer = build_scorer(
        judge_model=args.judge,
        categorical=args.categorical,
        graded=args.graded,
    )

    if args.arm == "all":
        arms = list_arms(scoring_data)
        log(f"\nProcessing arms: {arms}")
        rescores = score_all_arms(scorer, scoring_data, greedy, log_fn=log)
    else:
        log(f"\nProcessing arm: {args.arm}")
        rescores = {args.arm: score_arm(scorer, args.arm, scoring_data, greedy, log_fn=log)}

    questions = list(args.categorical) + list(args.graded)
    log("\nResults:")
    log(f"  questions: {questions}")
    summary: dict[str, dict[str, list[float]]] = {}
    for arm, rs in rescores.items():
        cores = compute_cores(rs)
        summary[arm] = cores
        log(f"\n  arm = {arm}  (n={rs.n_samples})")
        for name, core in cores.items():
            log(f"    {name:>22s}: {[round(v, 3) for v in core]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "questions": questions,
                    "categorical": args.categorical,
                    "graded": args.graded,
                    "cores": summary,
                },
                f,
                indent=2,
            )
        log(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
