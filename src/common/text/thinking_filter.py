"""Utilities for filtering thinking blocks from text."""

from __future__ import annotations

import re


def strip_thinking_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from text.

    Args:
        text: Text that may contain thinking blocks

    Returns:
        Text with thinking blocks removed
    """
    # Match <think>...</think> blocks (including newlines)
    pattern = r"<think>.*?</think>\s*"
    return re.sub(pattern, "", text, flags=re.DOTALL)
