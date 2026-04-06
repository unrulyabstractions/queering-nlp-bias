#!/usr/bin/env python3
"""Score, estimate, and visualize from an existing generation output.

Runs the three downstream pipeline stages given an already-generated output file:
    1. Score trajectories against scoring structures
    2. Estimate normativity from scores
    3. Generate visualizations

Usage:
    uv run python scripts/score_estimate_visualize.py out/<method>/generation.json trials/scoring/test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from estimate_normativity import estimate_normativity
from score_trajectories import score_trajectories

from src.common.experiment_types import OutputPaths
from src.common.logging import log_stage
from src.estimation import EstimationOutput, ScoringData
from src.estimation.estimation_experiment_types import EstimationResult
from src.estimation.logging.estimation_comparison_logging import log_setup_summary
from src.scoring import GenerationOutputData, ScoringConfig, ScoringOutput
from src.viz import visualize_result


def run_score_estimate_visualize(gen_json_path: Path, scoring_config_path: Path) -> None:
    """Run scoring, estimation, and visualization on an existing generation output."""
    scoring_out_path = ScoringOutput.compute_output_path(gen_json_path, scoring_config_path)
    estimation_out_path = EstimationOutput.compute_output_path(scoring_out_path)

    paths = OutputPaths(
        generation=gen_json_path,
        judgment=scoring_out_path,
        estimation=estimation_out_path,
    )

    log_stage(1, 3, "SCORE")
    scoring_cfg = ScoringConfig.load(scoring_config_path)
    gen_data = GenerationOutputData.load(gen_json_path)
    score_trajectories(scoring_cfg, scoring_config_path, gen_data, gen_json_path)

    log_stage(2, 3, "ESTIMATE")
    judgment_data = ScoringData.load(scoring_out_path)
    estimate_normativity(judgment_data, scoring_out_path)

    log_stage(3, 3, "VISUALIZE")
    method_name = gen_json_path.parent.name
    result = EstimationResult.from_estimation_file(method_name, paths)
    log_setup_summary(paths)
    visualize_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score, estimate, and visualize from an existing generation output"
    )
    parser.add_argument("generation_output", help="Path to generation output JSON")
    parser.add_argument("scoring_config", help="Path to scoring config JSON")
    args = parser.parse_args()

    gen_json_path = Path(args.generation_output)
    scoring_config_path = Path(args.scoring_config)

    if not gen_json_path.exists():
        print(f"Error: generation output not found: {gen_json_path}", file=sys.stderr)
        sys.exit(1)

    if not scoring_config_path.exists():
        print(f"Error: scoring config not found: {scoring_config_path}", file=sys.stderr)
        sys.exit(1)

    run_score_estimate_visualize(gen_json_path, scoring_config_path)


if __name__ == "__main__":
    main()
