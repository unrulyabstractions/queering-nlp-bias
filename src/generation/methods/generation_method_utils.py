"""Shared utilities for generation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.common.experiment_types import GenerationArm
    from src.inference import ModelRunner

    from ..generation_config import GenerationConfig


def compute_arm_token_lengths(
    runner: ModelRunner,
    config: GenerationConfig,
    arms: list[GenerationArm],
) -> list[int]:
    """Compute the token length for each arm (prompt + prefill).

    Args:
        runner: Model runner for tokenization
        config: Generation config with prompt
        arms: List of arms to compute lengths for

    Returns:
        List of token lengths, one per arm
    """
    formatted_prompt = runner.apply_chat_template(config.prompt)
    return [
        len(runner.encode_ids(formatted_prompt + arm.prefill, add_special_tokens=True))
        for arm in arms
    ]
