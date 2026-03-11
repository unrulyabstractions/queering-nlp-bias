"""Generate trajectories using forking paths algorithm.

Probes local branches around a greedy path by exploring one-step
deviations at positions where alternative tokens have high probability
and the model shows sufficient uncertainty (entropy).

Usage:
    python scripts/generate_by_forking_paths.py trials/generation/<config>.json
    python scripts/generate_by_forking_paths.py trials/generation/<config>.json \
        --max-alternates-per-position 5 \
        --min-prob-for-alternate 0.05 \
        --min-entropy-to-fork 1.0 \
        --samples-per-fork 2
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.script_utils import (
    ArgSpec,
    load_model,
    log_prompt_header,
    parse_generation_args,
)

from src.common.log import log, log_section
from src.generation import (
    ForkingParams,
    GenerationConfig,
    GenerationOutput,
    run_generation_pipeline,
)
from src.generation.methods.logging import log_tree_trajectories


def generate_by_forking_paths(
    config: GenerationConfig,
    config_path: Path,
    params: ForkingParams,
) -> None:
    """Run forking paths generation pipeline."""
    runner = load_model(config)

    log_section("Forking Paths Algorithm")
    params.print()

    # Show prompt structure
    log_prompt_header(config.prompt, config.trunk, config.branches)

    # Run generation pipeline with logging
    result = run_generation_pipeline(runner, config, method="forking-paths", log_fn=log)

    # Log trajectory details
    log_tree_trajectories(result.result, runner)

    # Save outputs
    n_trajs = len(result.result.trajectories)
    output_path = GenerationOutput.compute_output_path(config_path, method="forking-paths")
    result.output.save(output_path)
    log(f"\nSaved {n_trajs} trajectories to {output_path}")

    summary_path = GenerationOutput.compute_summary_path(config_path, method="forking-paths")
    result.output.save_summary(summary_path)
    log(f"Saved summary to {summary_path}")

    # Show summary
    result.output.summarize()


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories using forking paths algorithm",
        examples=[
            "config.json",
            "config.json --max-alternates-per-position 5 --min-prob-for-alternate 0.1",
            "config.json --min-entropy-to-fork 1.5 --samples-per-fork 3",
        ],
        extra_args=[
            ArgSpec(
                "max-alternates-per-position",
                int,
                "K",
                "Max alternate tokens per position",
            ),
            ArgSpec(
                "min-prob-for-alternate",
                float,
                "P",
                "Minimum probability for alternate token",
            ),
            ArgSpec(
                "min-entropy-to-fork", float, "H", "Minimum entropy to consider forking"
            ),
            ArgSpec("samples-per-fork", int, "N", "Continuations per fork point"),
        ],
    )

    generate_by_forking_paths(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.get_params("forking-paths"),
    )


if __name__ == "__main__":
    main()
