"""Logging utilities for generation methods.

This module provides logging functions for trajectory generation output.
"""

from __future__ import annotations

from src.common.logging import fmt_prob, log, log_section
from src.common.viz_utils import preview
from src.generation.generation_types import ArmGenerationResult
from src.inference import ModelRunner


def log_tree_trajectories(result: ArmGenerationResult, runner: ModelRunner) -> None:
    """Log trajectory texts and conditional probabilities.

    Args:
        result: Generation result with trajectories
        runner: Model runner for text decoding
    """
    prompt_len = result.prompt_length
    trunk_len = result.trunk_length

    log_section("Building Tree")

    # Table 1: Trajectory continuations (not full text including prompt)
    log(f"  Trajectories ({len(result.trajectories)} total):")
    log(f"  {'#':>3}  {'branch':<10} continuation")
    log("  " + "─" * 70)

    for i, traj in enumerate(result.trajectories):
        arm_index = result.arm_indices[i]
        display = "trunk" if arm_index == 0 else f"branch_{arm_index}"
        # Show only continuation (tokens after trunk), not full text
        continuation_ids = traj.token_ids[trunk_len:]
        continuation_text = runner.decode_ids(continuation_ids)
        # Show more text before cutting off (80 chars instead of 50)
        log(f"  {i:>3}  {display:<10} {preview(continuation_text, 80)}")
    log("")

    # Table 2: Conditional probabilities
    log(
        f"  Conditional probabilities (prompt_len={prompt_len}, trunk_len={trunk_len}):"
    )
    log(
        f"  {'#':>3}  {'branch':<10} {'p(t|prompt)':>11}  "
        f"{'p(t|trunk)':>11}  {'p(t|branch)':>11}  {'Finished?':>9}"
    )
    log("  " + "─" * 67)

    # Get EOS token from runner
    eos_token = runner.eos_token
    eos_token_id = runner.eos_token_id

    for i, traj in enumerate(result.trajectories):
        arm_index = result.arm_indices[i]
        p_prompt = traj.get_conditional_prob(prompt_len, traj.length) or 0.0

        if arm_index == 0:
            p_trunk = traj.get_conditional_prob(trunk_len, traj.length) or 0.0
            p_branch = p_trunk
        else:
            p_trunk = traj.get_conditional_prob(trunk_len - 1, traj.length) or 0.0
            p_branch = traj.get_conditional_prob(trunk_len, traj.length) or 0.0

        # Check if trajectory has EOS token (by token ID or text)
        continuation_ids = traj.token_ids[trunk_len:]
        is_finished = eos_token_id is not None and eos_token_id in continuation_ids
        if not is_finished and eos_token:
            continuation_text = runner.decode_ids(continuation_ids)
            is_finished = eos_token in continuation_text
        finished_str = "YES" if is_finished else "NO"

        display = "trunk" if arm_index == 0 else f"branch_{arm_index}"
        log(
            f"  {i:>3}  {display:<10} {fmt_prob(p_prompt, 11)}  "
            f"{fmt_prob(p_trunk, 11)}  {fmt_prob(p_branch, 11)}  {finished_str:>9}"
        )
    log("")
