#!/usr/bin/env python3
"""Run full experiment pipeline: generate -> score -> estimate.

Orchestrates the three-stage pipeline:
    1. Generate trajectories (simple-sampling, forking-paths, or seeking-entropy)
    2. Score trajectories against scoring structures
    3. Estimate normativity from scores

By default, runs ALL generation methods and compares results.

Usage:
    python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --forking-paths trials/generation/test.json trials/scoring/test.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from estimate_normativity import estimate_normativity
from generate_by_forking_paths import generate_by_forking_paths
from generate_by_seeking_entropy import generate_by_seeking_entropy
from generate_by_simple_sampling import generate_by_simple_sampling
from schemas import (
    EstimationOutput,
    GenerationConfig,
    GenerationOutput,
    GenerationOutputData,
    JudgmentData,
    JudgmentOutput,
    OutputPaths,
    ScoringConfig,
)
from schemas.experiment import (
    ExperimentResult,
    GenerationMethod,
    display_comparison,
    log_setup_summary,
)
from schemas.script_utils import (
    STAGE_GAP,
    log_experiment_start,
    log_header,
    log_major,
    log_stage,
)
from score_trajectories import score_trajectories

from src.common.seed import set_seed

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

ALL_METHODS: list[GenerationMethod] = [
    "simple-sampling",
    "forking-paths",
    "seeking-entropy",
]

METHOD_KEYWORDS: dict[GenerationMethod, str] = {
    "simple-sampling": "sampling",
    "forking-paths": "forking",
    "seeking-entropy": "entropy",
}


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedExperimentArgs:
    """Result of parsing experiment arguments."""

    gen_config_path: Path
    scoring_config_path: Path
    methods: list[GenerationMethod]
    overrides: dict[str, Any]


def parse_experiment_args() -> ParsedExperimentArgs:
    """Parse command-line arguments for experiment."""
    parser = argparse.ArgumentParser(
        description="Run full experiment: generate -> score -> estimate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("generation_config", help="Path to generation config JSON")
    parser.add_argument("scoring_config", help="Path to scoring config JSON")

    # Generation method (mutually exclusive, default = all)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--all", action="store_true", help="Run all methods (default)"
    )
    method_group.add_argument(
        "--simple-sampling", action="store_true", help="Simple sampling only"
    )
    method_group.add_argument(
        "--forking-paths", action="store_true", help="Forking paths only"
    )
    method_group.add_argument(
        "--seeking-entropy", action="store_true", help="Entropy-seeking only"
    )

    # Method-specific parameters
    parser.add_argument("--samples-per-branch", type=int, metavar="N")
    parser.add_argument("--max-alternates-per-position", type=int, metavar="K")
    parser.add_argument("--min-prob-for-alternate", type=float, metavar="P")
    parser.add_argument("--min-entropy-to-fork", type=float, metavar="H")
    parser.add_argument("--samples-per-fork", type=int, metavar="N")
    parser.add_argument("--samples-per-expansion", type=int, metavar="N")
    parser.add_argument("--num-expansion-rounds", type=int, metavar="K")

    args = parser.parse_args()

    # Determine methods
    if args.simple_sampling:
        methods: list[GenerationMethod] = ["simple-sampling"]
    elif args.forking_paths:
        methods = ["forking-paths"]
    elif args.seeking_entropy:
        methods = ["seeking-entropy"]
    elif args.all:
        methods = list(ALL_METHODS)
    else:
        # Default: only simple sampling
        methods = ["simple-sampling"]

    # Collect overrides
    overrides = {
        "samples_per_branch": args.samples_per_branch,
        "max_alternates_per_position": args.max_alternates_per_position,
        "min_prob_for_alternate": args.min_prob_for_alternate,
        "min_entropy_to_fork": args.min_entropy_to_fork,
        "samples_per_fork": args.samples_per_fork,
        "samples_per_expansion": args.samples_per_expansion,
        "num_expansion_rounds": args.num_expansion_rounds,
    }

    return ParsedExperimentArgs(
        gen_config_path=Path(args.generation_config),
        scoring_config_path=Path(args.scoring_config),
        methods=methods,
        overrides=overrides,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


def step_generate(
    config: GenerationConfig,
    config_path: Path,
    method: GenerationMethod,
) -> None:
    """Generate trajectories using the specified method."""
    log_stage(1, 3, f"GENERATE ({method})")

    if method == "forking-paths":
        generate_by_forking_paths(config, config_path, config.forking_params)
    elif method == "seeking-entropy":
        generate_by_seeking_entropy(config, config_path, config.entropy_params)
    else:
        generate_by_simple_sampling(config, config_path, config.sampling_params)


def step_score(scoring_path: Path, gen_output_path: Path) -> None:
    """Score generated trajectories."""
    log_stage(2, 3, "SCORE")

    scoring_cfg = ScoringConfig.load(scoring_path)
    gen_data = GenerationOutputData.load(gen_output_path)
    score_trajectories(scoring_cfg, scoring_path, gen_data, gen_output_path)


def step_estimate(judgment_path: Path) -> None:
    """Estimate normativity from judgments."""
    log_stage(3, 3, "ESTIMATE")

    judgment_data = JudgmentData.load(judgment_path)
    estimate_normativity(judgment_data, judgment_path)


# ══════════════════════════════════════════════════════════════════════════════
# Path Computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_paths(
    gen_config: Path,
    scoring_config: Path,
    method: GenerationMethod,
) -> OutputPaths:
    """Compute output paths for all pipeline stages."""
    keyword = METHOD_KEYWORDS.get(method, "sampling")
    gen_out = GenerationOutput.compute_output_path(gen_config, method=keyword)
    judge_out = JudgmentOutput.compute_output_path(gen_out, scoring_config)
    est_out = EstimationOutput.compute_output_path(judge_out)
    return OutputPaths(generation=gen_out, judgment=judge_out, estimation=est_out)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment Runners
# ══════════════════════════════════════════════════════════════════════════════


def run_single_experiment(
    gen_config_path: Path,
    scoring_config_path: Path,
    method: GenerationMethod,
    overrides: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run a single experiment with one generation method."""
    paths = compute_paths(gen_config_path, scoring_config_path, method)

    log_experiment_start(
        f"EXPERIMENT: {method}",
        paths,
        generation_config=gen_config_path,
        scoring_config=scoring_config_path,
        method=method,
    )

    # Load config and apply overrides
    config = GenerationConfig.load(gen_config_path)
    if overrides:
        config.apply_cli_overrides(overrides)
    set_seed(config.seed)

    # Run pipeline
    step_generate(config, gen_config_path, method)
    step_score(scoring_config_path, paths.generation)
    step_estimate(paths.judgment)

    # Load result and show setup summary
    result = ExperimentResult.from_estimation_file(method, paths)
    log_setup_summary(paths)

    return result


def run_all_experiments(args: ParsedExperimentArgs) -> list[ExperimentResult]:
    """Run experiments for multiple generation methods and compare."""
    if len(args.methods) > 1:
        log_major("MULTI-METHOD EXPERIMENT", f"Methods: {', '.join(args.methods)}")

    results = [
        run_single_experiment(
            args.gen_config_path,
            args.scoring_config_path,
            method,
            args.overrides,
        )
        for method in args.methods
    ]

    if len(results) > 1:
        display_comparison(results, gap=STAGE_GAP)

    log_header("EXPERIMENT COMPLETE", gap=1)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point."""
    args = parse_experiment_args()
    run_all_experiments(args)


if __name__ == "__main__":
    main()
