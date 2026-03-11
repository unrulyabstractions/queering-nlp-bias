#!/usr/bin/env python3
"""Run full experiment pipeline: generate -> score -> estimate.

Orchestrates the three-stage pipeline:
    1. Generate trajectories (any registered method)
    2. Score trajectories against scoring structures
    3. Estimate normativity from scores

Usage:
    python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --method forking trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --all trials/generation/test.json trials/scoring/test.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from estimate_normativity import estimate_normativity
from schemas.script_utils import (
    apply_cli_overrides_to_config,
    load_model,
    log_experiment_start,
    log_prompt_header,
)
from score_trajectories import score_trajectories

from src.common.log import log, log_section
from src.common.log_utils import (
    STAGE_GAP,
    log_header,
    log_major,
    log_stage,
)
from src.common.seed import set_seed
from src.estimation import EstimationOutput, ScoringData
from src.estimation.estimation_experiment_types import EstimationResult
from src.estimation.logging.estimation_comparison_logging import (
    display_comparison,
    log_setup_summary,
)
from src.generation import (
    GenerationConfig,
    GenerationOutput,
    OutputPaths,
    get_method_name_from_output,
    get_output_name,
    list_output_names,
    run_generation_pipeline,
)
from src.generation.methods.logging import log_tree_trajectories
from src.scoring import GenerationOutputData, ScoringConfig, ScoringOutput
from src.viz import visualize_generation_comparison, visualize_result

# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParsedExperimentArgs:
    """Result of parsing experiment arguments."""

    gen_config_path: Path
    scoring_config_path: Path
    methods: list[str]  # Method names from registry
    overrides: dict[str, Any]


def parse_experiment_args() -> ParsedExperimentArgs:
    """Parse command-line arguments for experiment."""
    available_output_names = list_output_names()

    parser = argparse.ArgumentParser(
        description="Run full experiment: generate -> score -> estimate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available methods: {', '.join(available_output_names)}",
    )
    parser.add_argument("generation_config", help="Path to generation config JSON")
    parser.add_argument("scoring_config", help="Path to scoring config JSON")

    # Method selection - uses full output names like "forking-paths"
    parser.add_argument(
        "--method",
        choices=available_output_names,
        default="simple-sampling",
        help="Generation method (default: simple-sampling)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available methods and compare",
    )

    # Method-specific parameters (apply to whichever method uses them)
    parser.add_argument("--samples-per-arm", type=int, metavar="N")
    parser.add_argument("--max-alternates-per-position", type=int, metavar="K")
    parser.add_argument("--min-prob-for-alternate", type=float, metavar="P")
    parser.add_argument("--min-entropy-to-fork", type=float, metavar="H")
    parser.add_argument("--samples-per-fork", type=int, metavar="N")
    parser.add_argument("--samples-per-expansion", type=int, metavar="N")
    parser.add_argument("--num-expansion-rounds", type=int, metavar="K")

    args = parser.parse_args()

    # Determine methods - convert output names to internal method names
    if args.all:
        methods = [get_method_name_from_output(name) for name in available_output_names]
    else:
        methods = [get_method_name_from_output(args.method)]

    # Collect overrides
    overrides = {
        "samples_per_arm": args.samples_per_arm,
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
    method: str,
) -> GenerationOutput:
    """Generate trajectories using the specified method."""
    output_name = get_output_name(method)
    log_stage(1, 3, f"GENERATE ({output_name})")

    runner = load_model(config)
    log_section(f"{output_name.replace('-', ' ').title()}")
    config.get_params(method).print()
    log_prompt_header(config.prompt, config.trunk, config.branches)

    result = run_generation_pipeline(
        runner=runner,
        config=config,
        method=method,
        log_fn=log,
    )

    log_tree_trajectories(result.result, runner)

    # Save output
    output_path = GenerationOutput.compute_output_path(config_path, method=method)
    result.output.save(output_path)
    log(f"\nSaved: {output_path}")

    return result.output


def step_score(
    scoring_path: Path,
    gen_output_path: Path,
) -> None:
    """Score generated trajectories."""
    log_stage(2, 3, "SCORE")

    scoring_cfg = ScoringConfig.load(scoring_path)
    gen_data = GenerationOutputData.load(gen_output_path)
    score_trajectories(scoring_cfg, scoring_path, gen_data, gen_output_path)


def step_estimate(judgment_path: Path) -> None:
    """Estimate normativity from judgments."""
    log_stage(3, 3, "ESTIMATE")

    judgment_data = ScoringData.load(judgment_path)
    estimate_normativity(judgment_data, judgment_path)


# ══════════════════════════════════════════════════════════════════════════════
# Path Computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_paths(
    gen_config: Path,
    scoring_config: Path,
    method: str,
) -> OutputPaths:
    """Compute output paths for all pipeline stages."""
    gen_out = GenerationOutput.compute_output_path(gen_config, method=method)
    judge_out = ScoringOutput.compute_output_path(gen_out, scoring_config)
    est_out = EstimationOutput.compute_output_path(judge_out)
    return OutputPaths(generation=gen_out, judgment=judge_out, estimation=est_out)


# ══════════════════════════════════════════════════════════════════════════════
# Experiment Runners
# ══════════════════════════════════════════════════════════════════════════════


def run_single_experiment(
    gen_config_path: Path,
    scoring_config_path: Path,
    method: str,
    overrides: dict[str, Any] | None = None,
) -> EstimationResult:
    """Run a single experiment with one generation method."""
    output_name = get_output_name(method)
    paths = compute_paths(gen_config_path, scoring_config_path, method)

    log_experiment_start(
        f"EXPERIMENT: {output_name}",
        paths,
        generation_config=gen_config_path,
        scoring_config=scoring_config_path,
        method=output_name,
    )

    # Load config and apply overrides
    config = GenerationConfig.load(gen_config_path)
    if overrides:
        apply_cli_overrides_to_config(config, overrides)
    set_seed(config.seed)

    # Run pipeline
    step_generate(config, gen_config_path, method)
    step_score(scoring_config_path, paths.generation)
    step_estimate(paths.judgment)

    # Load result and show setup summary
    result = EstimationResult.from_estimation_file(output_name, paths)
    log_setup_summary(paths)

    # Generate visualizations
    visualize_result(result)

    return result


def run_all_experiments(args: ParsedExperimentArgs) -> list[EstimationResult]:
    """Run experiments for multiple generation methods and compare."""
    if len(args.methods) > 1:
        method_names = [get_output_name(m) for m in args.methods]
        log_major("MULTI-METHOD EXPERIMENT", f"Methods: {', '.join(method_names)}")

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
        visualize_generation_comparison(results)

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
