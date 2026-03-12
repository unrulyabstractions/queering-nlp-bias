"""Just greedy generation method.

This module implements the simplest generation method: one greedy
trajectory per arm (temperature=0).

Algorithm:
    For each arm:
        1. Construct prompt with arm prefill
        2. Generate one trajectory with temperature=0 (greedy decoding)
        3. Return the single trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.common.viz_utils import preview
from src.inference import ModelRunner
from src.inference.generated_trajectory import GeneratedTrajectory

from ..generation_config import GenerationConfig
from ..generation_method_registry import GenerationMethodParams, register_method
from src.common.experiment_types import ArmGenerationResult

from .generation_method_utils import compute_arm_token_lengths

from .logging.gen_logging_utils import log_arm_header


@dataclass
class JustGreedyParams(GenerationMethodParams):
    """Parameters for just-greedy generation.

    No configurable parameters - always generates one greedy trajectory per arm.
    """

    name: ClassVar[str] = "just-greedy"


@register_method(JustGreedyParams)
def generate_just_greedy(
    runner: ModelRunner,
    config: GenerationConfig,
    params: JustGreedyParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate one greedy trajectory per arm (temperature=0).

    Args:
        runner: Model runner for generation
        config: Generation config with prompt, arms, and parameters
        params: Just-greedy parameters (currently empty)
        log_fn: Optional logging callback

    Returns:
        ArmGenerationResult containing one trajectory per arm
    """
    arms = config.get_arms(runner.skip_thinking_prefix)
    arm_token_lengths = compute_arm_token_lengths(runner, config, arms)

    all_trajectories: list[GeneratedTrajectory] = []
    all_arm_indices: list[int] = []

    for arm_idx, arm in enumerate(arms):
        if log_fn:
            log_arm_header(arm, log_fn)
            log_fn("\n  Generating greedy trajectory (temperature=0)...")

        # Generate single greedy trajectory
        traj = runner.generate_trajectory_from_prompt(
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=0.0,  # Greedy decoding
            prefilling=arm.prefill,
        )

        if log_fn:
            log_fn(f'    "{preview(traj.continuation_text, 70)}"')

        # Free heavy data (full_logits) immediately to reduce peak memory
        traj.pop_heavy()
        all_trajectories.append(traj)
        all_arm_indices.append(arm_idx)

    if log_fn:
        log_fn(f"\n  Summary: {len(all_trajectories)} greedy trajectories generated")

    return ArmGenerationResult(
        trajectories=all_trajectories,
        arm_indices=all_arm_indices,
        arm_token_lengths=arm_token_lengths,
        arms=arms,
    )
