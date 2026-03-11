"""Generate trajectories using simple temperature sampling.

Usage:
    python scripts/generate_by_simple_sampling.py trials/generation/<config>.json
    python scripts/generate_by_simple_sampling.py trials/generation/<config>.json \
        --samples-per-arm 5

Outputs:
    out/gen_sampling_<config>.json
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
    GenerationConfig,
    GenerationOutput,
    SamplingParams,
    run_generation_pipeline,
)
from src.generation.methods.logging import log_tree_trajectories


def generate_by_simple_sampling(
    config: GenerationConfig,
    config_path: Path,
    params: SamplingParams,
) -> None:
    """Run simple sampling generation pipeline."""
    runner = load_model(config)

    log_section("Simple Sampling")
    params.print()

    # Show prompt structure
    log_prompt_header(config.prompt, config.trunk, config.branches)

    # Run generation pipeline with logging
    result = run_generation_pipeline(runner, config, method="simple-sampling", log_fn=log)

    # Log trajectory details
    log_tree_trajectories(result.result, runner)

    # Save outputs
    n_trajs = len(result.result.trajectories)
    output_path = GenerationOutput.compute_output_path(config_path, method="simple-sampling")
    result.output.save(output_path)
    log(f"\nSaved {n_trajs} trajectories to {output_path}")

    summary_path = GenerationOutput.compute_summary_path(config_path, method="simple-sampling")
    result.output.save_summary(summary_path)
    log(f"Saved summary to {summary_path}")

    # Show summary
    result.output.summarize()


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories using simple temperature sampling",
        examples=["config.json", "config.json --samples-per-arm 10"],
        extra_args=[
            ArgSpec("samples-per-arm", int, "N", "Trajectories per arm"),
        ],
    )

    generate_by_simple_sampling(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.get_params("simple-sampling"),
    )


if __name__ == "__main__":
    main()
