"""Text-based visualization utilities for console output."""

from __future__ import annotations

import math


# ══════════════════════════════════════════════════════════════════════════════
# Text Formatting
# ══════════════════════════════════════════════════════════════════════════════


def escape_newlines(text: str) -> str:
    """Escape newlines for single-line display."""
    return text.replace("\n", "\\n")


def truncate(text: str, max_len: int = 50, suffix: str = "...") -> str:
    """Truncate text to max_len, adding suffix if truncated."""
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


def preview(text: str, max_len: int = 50) -> str:
    """Create a preview of text, escaping newlines and truncating if needed."""
    # Escape newlines for display
    escaped = text.replace("\n", "\\n")
    return truncate(escaped, max_len, "...")


def wrap_text(text: str, width: int = 78, indent: str = "  ") -> list[str]:
    """Wrap text to width with indent prefix on each line."""
    words = text.split()
    if not words:
        return []

    lines = []
    line = indent
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = indent + word
        else:
            line = line + " " + word if line != indent else indent + word
    if line.strip():
        lines.append(line)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Numeric Utilities
# ══════════════════════════════════════════════════════════════════════════════


def sanitize_float(x: float) -> float:
    """Replace inf/nan with finite values."""
    if math.isnan(x):
        return 0.0
    if math.isinf(x):
        return -1000.0 if x < 0 else 1000.0
    return x


def sanitize_floats(values: list[float]) -> list[float]:
    """Sanitize a list of floats."""
    return [sanitize_float(x) for x in values]


