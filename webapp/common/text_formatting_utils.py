"""Shared text formatting utilities for logging and display."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TextComponents:
    """Clean separation of prefill, generated, and full text."""

    prefill: str
    generated: str

    @property
    def full(self) -> str:
        """Combine prefill + generated with proper spacing."""
        if not self.prefill:
            return self.generated
        if not self.generated:
            return self.prefill
        # Add space if needed between prefill and generated
        if self.prefill.endswith((' ', '\n')) or self.generated.startswith((' ', '\n')):
            return self.prefill + self.generated
        return self.prefill + ' ' + self.generated


def combine_prefill_generated(prefill: str, generated: str) -> str:
    """Combine prefill and generated text with proper spacing."""
    return TextComponents(prefill, generated).full


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
