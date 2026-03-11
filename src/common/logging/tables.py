"""Table formatting utilities for logging."""

from __future__ import annotations

from .core import log
from .formatting import pad_left, pad_right


def log_table_header(
    columns: list[tuple[str, int, str]],
    indent: str = "  ",
    divider_width: int = 62,
) -> None:
    """Log a table header row with column formatting.

    Args:
        columns: List of (label, width, align) where align is '<', '>', or '^'
        indent: Indentation prefix
        divider_width: Width of divider line
    """
    parts = []
    for label, width, align in columns:
        if align == "<":
            parts.append(pad_right(label, width))
        elif align == ">":
            parts.append(pad_left(label, width))
        else:
            parts.append(label.center(width))
    log(indent + "  ".join(parts))
    log(indent + "-" * divider_width)


def log_table_row(
    cells: list[tuple[str, int, str]],
    indent: str = "  ",
) -> None:
    """Log a table row with column formatting.

    Args:
        cells: List of (value, width, align) where align is '<', '>', or '^'
        indent: Indentation prefix
    """
    parts = []
    for value, width, align in cells:
        if align == "<":
            parts.append(pad_right(value, width))
        elif align == ">":
            parts.append(pad_left(value, width))
        else:
            parts.append(value.center(width))
    log(indent + "  ".join(parts))
