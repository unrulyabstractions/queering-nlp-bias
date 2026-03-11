"""Simple temperature sampling generation method.

This module implements trajectory generation using simple temperature
sampling from a language model.

Algorithm:
    For each arm:
        1. Construct prompt with arm prefill
        2. Sample N trajectories using temperature sampling
        3. Collect all trajectories with arm indices
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from src.common.callback_types import LogFn
from src.common.default_config import SAMPLING_SAMPLES_PER_ARM
from src.common.viz_utils import preview
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

from ..generation_config import GenerationConfig
from ..generation_method_registry import GenerationMethodParams, register_method
from ..generation_types import ArmGenerationResult


@dataclass
class SamplingParams(GenerationMethodParams):
    """Parameters for simple temperature sampling."""

    samples_per_arm: int = field(default_factory=lambda: SAMPLING_SAMPLES_PER_ARM)

    name: ClassVar[str] = "simple-sampling"

    _cli_args: ClassVar[dict[str, str]] = {
        "samples_per_arm": "--samples-per-arm",
    }


def sample_from_arm(
    runner: ModelRunner,
    config: GenerationConfig,
    prefill: str,
    samples_per_arm: int,
    log_fn: LogFn | None = None,
) -> list[GeneratedTrajectory]:
    """Sample N trajectories for a single arm."""
    formatted_prompt = runner.apply_chat_template(config.prompt) + prefill

    trajectories = []
    for i in range(samples_per_arm):
        traj = runner.generate_trajectory_from_prompt(
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            prefilling=prefill,
        )

        if log_fn:
            text = runner.decode_ids(traj.token_ids)
            continuation = text[len(formatted_prompt) :]
            log_fn(f'    [{i + 1}/{samples_per_arm}] "{preview(continuation, 55)}"')

        # Free heavy data (full_logits) immediately to reduce peak memory
        traj.pop_heavy()
        trajectories.append(traj)

    return trajectories


@register_method(SamplingParams)
def generate_sampling(
    runner: ModelRunner,
    config: GenerationConfig,
    params: SamplingParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate trajectories using simple temperature sampling."""
    arms = config.get_arms(runner.skip_thinking_prefix)
    prompt_length = config.compute_prompt_length(runner)
    trunk_length = config.compute_trunk_length(runner)

    all_trajectories: list[GeneratedTrajectory] = []
    all_arm_indices: list[int] = []

    for arm in arms:
        if log_fn:
            if arm.name == "trunk":
                log_fn("\nTrunk")
                log_fn(f'  Continuation: "{config.trunk}"')
            else:
                branch_idx = arm.arm_index
                branch_text = (
                    config.branches[branch_idx - 1] if config.branches else arm.name
                )
                log_fn(f"\nBranch: branch_{branch_idx}")
                log_fn(f'  Continuation: "{config.trunk}{branch_text}"')
            log_fn(
                f"\n  Step 1: Sample trajectories ({params.samples_per_arm} samples)"
            )
            log_fn("  " + "─" * 50)

        trajectories = sample_from_arm(
            runner=runner,
            config=config,
            prefill=arm.prefill,
            samples_per_arm=params.samples_per_arm,
            log_fn=log_fn,
        )

        if log_fn:
            log_fn(f"\n  Summary: {params.samples_per_arm} trajectories generated")

        all_trajectories.extend(trajectories)
        all_arm_indices.extend(arm.arm_index for _ in trajectories)

    return ArmGenerationResult(
        trajectories=all_trajectories,
        arm_indices=all_arm_indices,
        trunk_length=trunk_length,
        prompt_length=prompt_length,
    )
