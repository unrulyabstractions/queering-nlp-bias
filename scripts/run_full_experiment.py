#!/usr/bin/env python3
"""Run full experiment pipeline: generate -> score -> estimate.

Orchestrates the three-stage pipeline:
    1. Generate trajectories (any registered method)
    2. Score trajectories against scoring structures
    3. Estimate normativity from scores

Optional: --dynamics computes drift and horizon dynamics for trajectories,
showing how deviance evolves through token positions.

Usage:
    python scripts/run_full_experiment.py trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --method forking trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --all trials/generation/test.json trials/scoring/test.json
    python scripts/run_full_experiment.py --dynamics trials/generation/test.json trials/scoring/test.json
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

from src.common.default_config import DYNAMICS_ARMS, DYNAMICS_STEP, DYNAMICS_TRAJS_PER_ARM
from src.common.logging import (
    STAGE_GAP,
    log,
    log_header,
    log_major,
    log_section,
    log_stage,
)
from src.common.profiler import P, profile
from src.common.random_seed import set_seed
from src.dynamics import compute_dynamics, plot_dynamics, save_dynamics_json
from src.estimation import EstimationOutput, ScoringData
from src.estimation.estimation_experiment_types import EstimationResult
from src.estimation.logging.estimation_comparison_logging import (
    display_comparison,
    log_setup_summary,
)
from src.common.experiment_types import OutputPaths
from src.generation import (
    GenerationConfig,
    GenerationOutput,
    get_method_name_from_output,
    get_output_name,
    list_output_names,
    run_generation_pipeline,
)
from src.generation.methods.logging import log_tree_trajectories
from src.inference import ModelRunner
from src.scoring import GenerationOutputData, ScoringConfig, ScoringOutput
from src.scoring.scorer import Scorer
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
    dynamics: bool  # Whether to compute drift/horizon dynamics
    profile: bool  # Whether to enable profiling
    base_dir: str  # Output base directory (out/ or generation_compare/)
    include_method_in_path: bool  # Whether to include method name in output path


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
    parser.add_argument(
        "--dynamics",
        action="store_true",
        help="Compute drift/horizon dynamics for trajectories",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling and print timing report",
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
        base_dir = "generation_compare"
        include_method_in_path = True  # Include method name when comparing multiple
    else:
        methods = [get_method_name_from_output(args.method)]
        base_dir = "out"
        include_method_in_path = False  # Simpler path for single method

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
        dynamics=args.dynamics,
        profile=args.profile,
        base_dir=base_dir,
        include_method_in_path=include_method_in_path,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ══════════════════════════════════════════════════════════════════════════════


@profile
def step_generate(
    config: GenerationConfig,
    config_path: Path,
    method: str,
    base_dir: str = "out",
    include_method: bool = False,
) -> GenerationOutput:
    """Generate trajectories using the specified method."""
    output_name = get_output_name(method)
    log_stage(1, 3, f"GENERATE ({output_name})")

    runner = load_model(config)
    log_section(f"{output_name.replace('-', ' ').title()}")
    config.get_params(method).print()
    log_prompt_header(config.prompt, config.trunk, config.branches, config.twig_variations)

    result = run_generation_pipeline(
        runner=runner,
        config=config,
        method=method,
        log_fn=log,
    )

    log_tree_trajectories(result.result, runner)

    # Save output (and copy original config)
    output_path = GenerationOutput.compute_output_path(
        config_path, method=method, base_dir=base_dir, include_method=include_method
    )
    result.output.save(output_path, config_path=config_path)
    log(f"\nSaved: {output_path}")

    # Save human-readable summary
    summary_path = GenerationOutput.compute_summary_path(
        config_path, method=method, base_dir=base_dir, include_method=include_method
    )
    result.output.save_summary(summary_path)
    log(f"Saved summary: {summary_path}")

    return result.output


@profile
def step_score(
    scoring_path: Path,
    gen_output_path: Path,
) -> None:
    """Score generated trajectories."""
    log_stage(2, 3, "SCORE")

    scoring_cfg = ScoringConfig.load(scoring_path)
    gen_data = GenerationOutputData.load(gen_output_path)
    score_trajectories(scoring_cfg, scoring_path, gen_data, gen_output_path)


@profile
def step_estimate(judgment_path: Path) -> None:
    """Estimate normativity from judgments."""
    log_stage(3, 3, "ESTIMATE")

    judgment_data = ScoringData.load(judgment_path)
    estimate_normativity(judgment_data, judgment_path)


def _apply_string_selection(text: str) -> str:
    """Apply default string selection (strip thinking blocks)."""
    from src.common.default_config import STRING_SELECTION

    if STRING_SELECTION == "NonThinkingContinuation":
        from src.common.text import strip_thinking_blocks
        return strip_thinking_blocks(text)
    return text


def _select_extremal_trajectories(all_trajs: list) -> list[tuple[int, str, str, int]]:
    """Select N most extremal trajectories per arm (alternating low/high inv_ppl)."""
    import math
    from itertools import groupby

    from src.estimation.arm_types import classify_arm

    def inv_ppl(t) -> float:
        lp = t.conditional_logprobs.get(t.arm, -1000.0)
        return math.exp(lp / t.n_generated_tokens) if t.n_generated_tokens > 0 else 0.0

    def pick_extremal(trajs: list, n: int) -> list:
        """Pick n most extremal from sorted list, alternating from ends."""
        result = []
        lo, hi = 0, len(trajs) - 1
        while len(result) < n and lo <= hi:
            result.append(trajs[lo])
            lo += 1
            if len(result) < n and lo <= hi:
                result.append(trajs[hi])
                hi -= 1
        return result

    # Filter to configured arm types
    filtered = [t for t in all_trajs if classify_arm(t.arm).value in DYNAMICS_ARMS]

    selected = []
    for _, group in groupby(sorted(filtered, key=lambda t: t.arm), key=lambda t: t.arm):
        by_ppl = sorted(group, key=inv_ppl)
        for t in pick_extremal(by_ppl, DYNAMICS_TRAJS_PER_ARM):
            # Apply string selection (e.g., strip thinking blocks)
            text = _apply_string_selection(t.text)
            selected.append((t.traj_idx, t.arm, text, t.n_generated_tokens))

    return selected


@profile
def step_dynamics(result: EstimationResult, scoring_config_path: Path) -> None:
    """Compute drift and horizon dynamics for trajectories."""
    log_section("DYNAMICS")
    log("Computing dynamics (pull, drift, horizon)...")

    scorer = Scorer.load(scoring_config_path)
    log(f"Loaded scorer: {scorer.num_structures} structures")

    # Load trajectories from scoring data
    scoring_data = ScoringData.load(result.paths.judgment)
    all_trajs = scoring_data.get_all_trajectories()

    # Select extremal trajectories (highest and lowest inv_ppl per arm)
    trajectories = _select_extremal_trajectories(all_trajs)

    log(f"Selected {len(trajectories)} trajectories (extremal inv_ppl per arm)")
    for traj_idx, arm, _, n_tokens in trajectories:
        log(f"  [{traj_idx}] {arm} ({n_tokens} tokens)")

    # Compute dynamics
    dynamics_result = compute_dynamics(trajectories, scorer, step=DYNAMICS_STEP, log_fn=log)

    # Save dynamics data as JSON
    dynamics_json_path = result.paths.estimation.parent / "dynamics.json"
    save_dynamics_json(dynamics_result, dynamics_json_path)
    log(f"Saved: {dynamics_json_path}")

    # Generate visualizations in viz/dynamics folder next to estimation.json
    output_dir = result.paths.estimation.parent / "viz" / "dynamics"
    saved_paths = plot_dynamics(dynamics_result, output_dir)

    for path in saved_paths:
        log(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Path Computation
# ══════════════════════════════════════════════════════════════════════════════


def compute_paths(
    gen_config: Path,
    scoring_config: Path,
    method: str,
    base_dir: str = "out",
    include_method: bool = False,
) -> OutputPaths:
    """Compute output paths for all pipeline stages."""
    gen_out = GenerationOutput.compute_output_path(
        gen_config, method=method, base_dir=base_dir, include_method=include_method
    )
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
    dynamics: bool = False,
    base_dir: str = "out",
    include_method: bool = False,
) -> EstimationResult:
    """Run a single experiment with one generation method."""
    output_name = get_output_name(method)
    paths = compute_paths(
        gen_config_path, scoring_config_path, method,
        base_dir=base_dir, include_method=include_method
    )

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
    step_generate(config, gen_config_path, method, base_dir=base_dir, include_method=include_method)
    step_score(scoring_config_path, paths.generation)
    step_estimate(paths.judgment)

    # Load result and show setup summary
    result = EstimationResult.from_estimation_file(output_name, paths)
    log_setup_summary(paths)

    # Generate visualizations
    visualize_result(result)

    # Compute dynamics if requested
    if dynamics:
        step_dynamics(result, scoring_config_path)

    return result


def run_all_experiments(args: ParsedExperimentArgs) -> list[EstimationResult]:
    """Run experiments for multiple generation methods and compare."""
    if len(args.methods) > 1:
        method_names = [get_output_name(m) for m in args.methods]
        log_major("MULTI-METHOD EXPERIMENT", f"Methods: {', '.join(method_names)}")
        log(f"Output directory: {args.base_dir}/")

    results = [
        run_single_experiment(
            args.gen_config_path,
            args.scoring_config_path,
            method,
            args.overrides,
            dynamics=args.dynamics,
            base_dir=args.base_dir,
            include_method=args.include_method_in_path,
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

    # Enable/disable profiling
    if args.profile:
        P.enable()
    else:
        P.disable()

    run_all_experiments(args)

    # Print profiling report if enabled
    if args.profile:
        P.report()


if __name__ == "__main__":
    main()
