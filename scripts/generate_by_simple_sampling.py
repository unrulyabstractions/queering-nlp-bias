#!/usr/bin/env python3
"""Generate trajectories using simple temperature sampling.

Usage:
    python scripts/generate_by_simple_sampling.py trials/generation/<config>.json
    python scripts/generate_by_simple_sampling.py trials/generation/<config>.json \
        --samples-per-branch 5

Outputs:
    out/gen_sampling_<config>.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import BranchGenerationResult, GenerationConfig, SamplingParams
from schemas.script_utils import (
    ArgSpec,
    build_and_save_tree,
    load_model,
    log_branch_header,
    log_step,
    parse_generation_args,
)

from src.common.log import log, log_section
from src.common.viz_utils import preview
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory


def sample_from_branch(
    runner: ModelRunner,
    config: GenerationConfig,
    branch_name: str,
    prefill: str,
    samples_per_branch: int,
) -> list[GeneratedTrajectory]:
    """Sample N trajectories for a single branch."""
    formatted_prompt = runner.apply_chat_template(config.prompt) + prefill
    log_branch_header(branch_name, formatted_prompt)

    log_step(1, "Sample trajectories", f"{samples_per_branch} samples")

    trajectories = []
    for i in range(samples_per_branch):
        traj = runner.generate_trajectory_from_prompt(
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            prefilling=prefill,
        )
        trajectories.append(traj)

        text = runner.decode_ids(traj.token_ids)
        continuation = text[len(formatted_prompt) :]

        log(f'    [{i + 1}/{samples_per_branch}] "{preview(continuation, 55)}"')

    log(f"  Summary: {samples_per_branch} trajectories generated", gap=1)

    return trajectories


def generate_for_all_branches(
    runner: ModelRunner,
    config: GenerationConfig,
    params: SamplingParams,
) -> BranchGenerationResult:
    """Generate trajectories for all branches."""
    branches = config.get_branches(runner.skip_thinking_prefix)
    prompt_length = config.compute_prompt_length(runner)
    trunk_length = config.compute_trunk_length(runner)

    all_trajectories: list[GeneratedTrajectory] = []
    all_group_indices: list[int] = []

    for branch in branches:
        trajectories = sample_from_branch(
            runner, config, branch.name, branch.prefill, params.samples_per_branch
        )
        all_trajectories.extend(trajectories)
        all_group_indices.extend(branch.group_idx for _ in trajectories)

    return BranchGenerationResult(
        trajectories=all_trajectories,
        group_indices=all_group_indices,
        trunk_length=trunk_length,
        prompt_length=prompt_length,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════


def generate_by_simple_sampling(
    config: GenerationConfig,
    config_path: Path,
    params: SamplingParams,
) -> None:
    """Run simple sampling generation pipeline."""
    runner = load_model(config)

    log_section("Simple Sampling")
    params.print()

    result = generate_for_all_branches(runner, config, params)

    build_and_save_tree(
        result=result,
        config=config,
        config_path=config_path,
        runner=runner,
        method="sampling",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parsed = parse_generation_args(
        description="Generate trajectories using simple temperature sampling",
        examples=["config.json", "config.json --samples-per-branch 10"],
        extra_args=[
            ArgSpec("samples-per-branch", int, "N", "Trajectories per branch"),
        ],
    )

    generate_by_simple_sampling(
        config=parsed.config,
        config_path=parsed.config_path,
        params=parsed.config.sampling_params,
    )


if __name__ == "__main__":
    main()
