"""Shared utilities for generation methods."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.common.experiment_types import GenerationArm
    from src.inference import ModelRunner

    from ..generation_config import GenerationConfig


def get_arm_prompt(arm: "GenerationArm", config: "GenerationConfig") -> str:
    """Return the prompt for a specific arm.

    In template mode each arm carries its own filled prompt; in traditional
    mode all arms share config.prompt.
    """
    return arm.prompt if arm.prompt else config.prompt


def compute_arm_token_lengths(
    runner: "ModelRunner",
    config: "GenerationConfig",
    arms: "list[GenerationArm]",
) -> list[int]:
    """Compute the token length for each arm (prompt + prefill).

    Args:
        runner: Model runner for tokenization
        config: Generation config with prompt
        arms: List of arms to compute lengths for

    Returns:
        List of token lengths, one per arm
    """
    return [
        len(
            runner.encode_ids(
                runner.apply_chat_template(get_arm_prompt(arm, config)) + arm.prefill,
                add_special_tokens=True,
            )
        )
        for arm in arms
    ]
