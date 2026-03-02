#!/usr/bin/env python3
"""Run full experiment pipeline: generate -> score -> estimate.

Orchestrates the three-stage pipeline:
    1. Generate trajectories (simple-sampling, forking-paths, or seeking-entropy)
    2. Score trajectories against scoring structures
    3. Estimate normativity from scores

By default, runs ALL generation methods and compares results.

Usage:
    # Run all methods (default) and compare:
    python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/test.json

    # Run a specific method:
    python scripts/run_full_experiment.py --forking-paths trials/generation/test.json trials/scoring/test.json

Outputs:
    out/gen_<method>_<name>.json
    out/score_<method>_<name>_<scoring>.json
    out/est_<method>_<name>_<scoring>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

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
from score_trajectories import score_trajectories

from src.common.log import log, log_params
from src.common.seed import set_seed

# ══════════════════════════════════════════════════════════════════════════════
# Types and Constants
# ══════════════════════════════════════════════════════════════════════════════

GenerationMethod = Literal["simple-sampling", "forking-paths", "seeking-entropy"]

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

HEADER_WIDTH = 60
STAGE_GAP = 4


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    method: GenerationMethod
    paths: OutputPaths
    core: list[float]
    core_inv_ppl: list[float]
    deviance_avg: float
    deviance_avg_inv_ppl: float
    n_trajectories: int


# ══════════════════════════════════════════════════════════════════════════════
# Logging Helpers
# ══════════════════════════════════════════════════════════════════════════════


def log_header(title: str, gap: int = 0) -> None:
    """Log a section header."""
    log("═" * HEADER_WIDTH, gap=gap)
    log(title)
    log("═" * HEADER_WIDTH)


def log_stage(step: int, total: int, title: str) -> None:
    """Log a pipeline stage separator."""
    log("", gap=STAGE_GAP)
    log("▓" * HEADER_WIDTH)
    log(f"▓  STAGE {step}/{total}: {title}")
    log("▓" * HEADER_WIDTH)


def log_output_paths(paths: OutputPaths, gap: int = 0) -> None:
    """Log output file paths."""
    log("Output files:", gap=gap)
    log(f"  -> {paths.generation}")
    log(f"  -> {paths.judgment}")
    log(f"  -> {paths.estimation}")


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


def get_method_keyword(method: GenerationMethod) -> str:
    """Get short keyword for filenames."""
    return METHOD_KEYWORDS.get(method, "sampling")


def compute_paths(
    gen_config: Path,
    scoring_config: Path,
    method: GenerationMethod,
) -> OutputPaths:
    """Compute output paths for all pipeline stages."""
    keyword = get_method_keyword(method)
    gen_out = GenerationOutput.compute_output_path(gen_config, method=keyword)
    judge_out = JudgmentOutput.compute_output_path(gen_out, scoring_config)
    est_out = EstimationOutput.compute_output_path(judge_out)
    return OutputPaths(generation=gen_out, judgment=judge_out, estimation=est_out)


# ══════════════════════════════════════════════════════════════════════════════
# Single Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════


def run_single_experiment(
    gen_config_path: Path,
    scoring_config_path: Path,
    method: GenerationMethod,
    overrides: dict[str, Any] | None = None,
) -> ExperimentResult:
    """Run a single experiment with one generation method.

    Returns:
        ExperimentResult with paths and core estimation summary.
    """
    paths = compute_paths(gen_config_path, scoring_config_path, method)

    # Log header
    log_header(f"EXPERIMENT: {method}")
    log_params(
        generation_config=gen_config_path,
        scoring_config=scoring_config_path,
        method=method,
    )
    log_output_paths(paths, gap=1)
    log("═" * HEADER_WIDTH)

    # Load config and apply overrides
    config = GenerationConfig.load(gen_config_path)
    if overrides:
        config.apply_cli_overrides(overrides)
    set_seed(config.seed)

    # Run pipeline
    step_generate(config, gen_config_path, method)
    step_score(scoring_config_path, paths.generation)
    step_estimate(paths.judgment)

    # Load estimation results for comparison
    with open(paths.estimation) as f:
        est_data = json.load(f)

    # Extract trunk group (index 0) statistics
    trunk = est_data["groups"][0]

    return ExperimentResult(
        method=method,
        paths=paths,
        core=trunk["core"],
        core_inv_ppl=trunk.get("core_inv_ppl", []),
        deviance_avg=trunk["deviance_avg"],
        deviance_avg_inv_ppl=trunk.get("deviance_avg_inv_ppl", 0.0),
        n_trajectories=len(trunk["trajectories"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Comparison Display
# ══════════════════════════════════════════════════════════════════════════════


def _fmt_core(core: list[float], max_items: int = 4) -> str:
    """Format core vector for display."""
    if not core:
        return "[]"
    items = ", ".join(f"{c:.3f}" for c in core[:max_items])
    if len(core) > max_items:
        return f"[{items}, ...]"
    return f"[{items}]"


def display_comparison(results: list[ExperimentResult]) -> None:
    """Display side-by-side comparison of estimation results."""
    log("", gap=STAGE_GAP)
    log("█" * HEADER_WIDTH)
    log("█  COMPARISON: Core Estimation by Generation Method")
    log("█" * HEADER_WIDTH)

    # Table header
    log("")
    log(f"  {'Method':<18} {'N':>4}  {'Core (prob-weighted)':<28} {'E[∂]':>8}")
    log("  " + "─" * 62)

    for r in results:
        core_str = _fmt_core(r.core)
        log(
            f"  {r.method:<18} {r.n_trajectories:>4}  {core_str:<28} {r.deviance_avg:>8.4f}"
        )

    # Inv-perplexity weighted comparison
    if any(r.core_inv_ppl for r in results):
        log("")
        log(f"  {'Method':<18} {'N':>4}  {'Core (inv-ppl-weighted)':<28} {'E[∂]':>8}")
        log("  " + "─" * 62)

        for r in results:
            core_str = _fmt_core(r.core_inv_ppl)
            log(
                f"  {r.method:<18} {r.n_trajectories:>4}  {core_str:<28} {r.deviance_avg_inv_ppl:>8.4f}"
            )

    log("")


# ══════════════════════════════════════════════════════════════════════════════
# Multi-Experiment Runner
# ══════════════════════════════════════════════════════════════════════════════


def run_all_experiments(
    gen_config_path: str,
    scoring_config_path: str,
    methods: list[GenerationMethod] | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[ExperimentResult]:
    """Run experiments for multiple generation methods and compare.

    Args:
        gen_config_path: Path to generation config
        scoring_config_path: Path to scoring config
        methods: Methods to run (default: all methods)
        overrides: CLI parameter overrides

    Returns:
        List of ExperimentResult for each method.
    """
    methods = methods or ALL_METHODS
    gen_path = Path(gen_config_path)
    scoring_path = Path(scoring_config_path)

    # Header for multi-experiment run
    if len(methods) > 1:
        log("█" * HEADER_WIDTH)
        log("█  MULTI-METHOD EXPERIMENT")
        log(f"█  Methods: {', '.join(methods)}")
        log("█" * HEADER_WIDTH)

    # Run each experiment
    results = []
    for method in methods:
        result = run_single_experiment(gen_path, scoring_path, method, overrides)
        results.append(result)

    # Display comparison if multiple methods
    if len(results) > 1:
        display_comparison(results)

    # Final summary
    log_header("EXPERIMENT COMPLETE", gap=1)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        description="Run full experiment: generate -> score -> estimate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("generation_config", help="Path to generation config JSON")
    parser.add_argument("scoring_config", help="Path to scoring config JSON")

    # Generation method (mutually exclusive, default = all)
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--all",
        action="store_true",
        help="Run all generation methods and compare (default)",
    )
    method_group.add_argument(
        "--simple-sampling",
        action="store_true",
        help="Use simple temperature sampling only",
    )
    method_group.add_argument(
        "--forking-paths",
        action="store_true",
        help="Use forking paths generation only",
    )
    method_group.add_argument(
        "--seeking-entropy",
        action="store_true",
        help="Use entropy-seeking generation only",
    )

    # Method-specific parameters
    parser.add_argument(
        "--samples-per-branch",
        type=int,
        metavar="N",
        help="[simple-sampling] Trajectories per branch",
    )
    parser.add_argument(
        "--max-alternates-per-position",
        type=int,
        metavar="K",
        help="[forking-paths] Max alternate tokens per position",
    )
    parser.add_argument(
        "--min-prob-for-alternate",
        type=float,
        metavar="P",
        help="[forking-paths] Min probability for alternates",
    )
    parser.add_argument(
        "--min-entropy-to-fork",
        type=float,
        metavar="H",
        help="[forking-paths] Min entropy at position to fork",
    )
    parser.add_argument(
        "--samples-per-fork",
        type=int,
        metavar="N",
        help="[forking-paths] Continuations per fork point",
    )
    parser.add_argument(
        "--samples-per-expansion",
        type=int,
        metavar="N",
        help="[seeking-entropy] Trajectories per expansion",
    )
    parser.add_argument(
        "--num-expansion-rounds",
        type=int,
        metavar="K",
        help="[seeking-entropy] Number of expansion rounds",
    )

    return parser


def get_methods_from_args(args: argparse.Namespace) -> list[GenerationMethod] | None:
    """Determine which methods to run from args. None means all."""
    if args.simple_sampling:
        return ["simple-sampling"]
    if args.forking_paths:
        return ["forking-paths"]
    if args.seeking_entropy:
        return ["seeking-entropy"]
    # Default: all methods
    return None


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect CLI parameter overrides."""
    return {
        "samples_per_branch": args.samples_per_branch,
        "max_alternates_per_position": args.max_alternates_per_position,
        "min_prob_for_alternate": args.min_prob_for_alternate,
        "min_entropy_to_fork": args.min_entropy_to_fork,
        "samples_per_fork": args.samples_per_fork,
        "samples_per_expansion": args.samples_per_expansion,
        "num_expansion_rounds": args.num_expansion_rounds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args()

    methods = get_methods_from_args(args)
    overrides = collect_overrides(args)

    run_all_experiments(
        gen_config_path=args.generation_config,
        scoring_config_path=args.scoring_config,
        methods=methods,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
