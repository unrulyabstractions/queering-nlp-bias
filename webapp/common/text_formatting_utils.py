"""Shared text formatting utilities for logging and display."""

from __future__ import annotations


def truncate_for_log(text: str, max_len: int = 80) -> str:
    """Truncate text for logging, adding ellipsis if needed.

    Replaces newlines with spaces and strips whitespace before truncating.
    """
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def format_scores(scores: list[float]) -> str:
    """Format a list of scores for logging."""
    return "[" + ", ".join(f"{s:.3f}" for s in scores) + "]"
