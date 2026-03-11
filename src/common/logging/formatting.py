"""Text formatting utilities for logging.

Provides consistent alignment, centering, and text formatting functions.
"""

from __future__ import annotations

import re

# Default display width for centering
DEFAULT_WIDTH = 76


def center(text: str, width: int = DEFAULT_WIDTH, fill: str = " ") -> str:
    """Center text within a given width.

    Args:
        text: Text to center
        width: Total width (default 76)
        fill: Fill character (default space)

    Returns:
        Centered string
    """
    return text.center(width, fill)


def center_block(lines: list[str], width: int = DEFAULT_WIDTH) -> list[str]:
    """Center a block of lines.

    Args:
        lines: List of text lines
        width: Total width

    Returns:
        List of centered lines
    """
    return [center(line, width) for line in lines]


def pad_left(text: str, width: int, fill: str = " ") -> str:
    """Right-align text (pad left).

    Args:
        text: Text to align
        width: Total width
        fill: Fill character

    Returns:
        Right-aligned string
    """
    return text.rjust(width, fill)


def pad_right(text: str, width: int, fill: str = " ") -> str:
    """Left-align text (pad right).

    Args:
        text: Text to align
        width: Total width
        fill: Fill character

    Returns:
        Left-aligned string
    """
    return text.ljust(width, fill)


def indent(text: str, spaces: int = 2) -> str:
    """Add indentation to text.

    Args:
        text: Text to indent
        spaces: Number of spaces to add

    Returns:
        Indented string
    """
    prefix = " " * spaces
    return prefix + text


def fmt_prob(p: float, width: int = 10) -> str:
    """Format probability, using scientific notation for very small values.

    Args:
        p: Probability value
        width: Minimum width

    Returns:
        Formatted probability string
    """
    if p < 0.0001:
        return f"{p:>{width}.1e}"
    return f"{p:>{width}.4f}"


def oneline(text: str) -> str:
    """Collapse whitespace to single spaces for display.

    Args:
        text: Text with possible multiple whitespace

    Returns:
        Single-line text with collapsed whitespace
    """
    return re.sub(r"\s+", " ", text).strip()


def preview(text: str, max_len: int = 50) -> str:
    """Truncate text for preview display.

    Args:
        text: Text to preview
        max_len: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    text = oneline(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
