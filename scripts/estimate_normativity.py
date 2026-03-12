"""Estimate normativity from scoring results.

Usage:
    python scripts/estimate_normativity.py out/<method>/score_<name>.json

Outputs:
    out/<method>/est_<name>.json

Computes structure-aware diversity metrics:
- Core: Expected system compliance (average scores)
- Orientation: Deviation from core per trajectory
- Deviance: Scalar non-normativity (orientation magnitude)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.logging import log, log_pipeline_header, log_step
from src.estimation import (
    EstimationOutput,
    ScoringData,
    run_estimation_pipeline,
)
from src.estimation.logging import (
    log_arm_statistics,
    log_continuations_by_branch,
    log_trajectories_with_scores,
)


def estimate_normativity(data: ScoringData, scores_path: Path) -> None:
    """Run normativity estimation pipeline with logging."""
    log_pipeline_header(
        "NORMATIVITY ESTIMATION",
        {
            "Input": str(scores_path),
            "Generation file": data.generation_file,
            "Judge model": data.judge_model,
            "Embedding model": data.embedding_model,
        },
    )

    # Step 0: Show continuations by branch
    log_continuations_by_branch(data)

    # Step 1: Show all trajectories with scores
    log_trajectories_with_scores(data)

    # Step 2: Show arm statistics
    by_arm = data.group_by_arm()
    log_arm_statistics(data, by_arm)

    # Run the estimation pipeline
    result = run_estimation_pipeline(data, str(scores_path))

    # Save outputs
    log_step(3, "Save output", str(EstimationOutput.compute_output_path(scores_path)))
    out_path = EstimationOutput.compute_output_path(scores_path)
    result.output.save(out_path)
    log(f"    Saved to {out_path}")

    log_step(4, "Save summary", str(EstimationOutput.compute_summary_path(scores_path)))
    summary_path = EstimationOutput.compute_summary_path(scores_path)
    result.output.save_summary(summary_path)
    log(f"    Saved to {summary_path}")

    # Show summary
    result.output.summarize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate normativity from scores")
    parser.add_argument("scores", help="Path to scoring output JSON")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    data = ScoringData.load(scores_path)

    estimate_normativity(data=data, scores_path=scores_path)


if __name__ == "__main__":
    main()
