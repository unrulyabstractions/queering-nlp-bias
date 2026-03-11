"""Generate trajectories by seeking high-entropy positions.

Iteratively expands a tree at positions where the model is most uncertain
(highest next-token entropy).

Usage:
    python scripts/generate_by_seeking_entropy.py trials/generation/<config>.json
    python scripts/generate_by_seeking_entropy.py trials/generation/<config>.json \
        --samples-per-expansion 3 \
        --num-expansion-rounds 4
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
    EntropySeekingParams,
    GenerationConfig,
    GenerationOutput,
    run_generation_pipeline,
)
from src.generation.methods.logging import log_tree_trajectories


def generate_by_seeking_entropy(
    config: GenerationConfig,
    config_path: Path,
    params: EntropySeekingParams,
) -> None:
    """Run entropy-seeking generation pipeline."""
    runner = load_model(config)

    log_section("Entropy-Seeking Algorithm")
    params.print()

    # Show prompt structure
    log_prompt_header(config.prompt, config.trunk, config.branches)

    # Run generation pipeline with logging
    result = run_generation_pipeline(runner, config, method="seeking-entropy", log_fn=log)

    # Log trajectory details
    log_tree_trajectories(result.result, runner)

    # Save outputs
    n_trajs = len(result.result.trajectories)
    output_path = GenerationOutput.compute_output_path(config_path, method="seeking-entropy")
    result.output.save(output_path)
    log(f"\nSaved {n_trajs} trajectories to {output_path}")

    summary_path = GenerationOutput.compute_summary_path(config_path, method="seeking-entropy")
    result.output.save_summary(summary_path)
    log(f"Saved summary to {summary_path}")

    # Show summary
    result.output.summarize()


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories by seeking high-entropy positions",
        examples=[
            "config.json",
            "config.json --samples-per-expansion 3 --num-expansion-rounds 5",
        ],
        extra_args=[
            ArgSpec("samples-per-expansion", int, "N", "Trajectories per expansion"),
            ArgSpec("num-expansion-rounds", int, "K", "Number of expansion rounds"),
        ],
    )

    generate_by_seeking_entropy(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.get_params("seeking-entropy"),
    )


if __name__ == "__main__":
    main()
