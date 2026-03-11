"""Centralized logging utilities for clean output formatting.

This module provides consistent logging utilities used across all scripts.
It has no dependencies on other schema modules to avoid circular imports.
"""

from __future__ import annotations

import re

from src.common.log import log

# ══════════════════════════════════════════════════════════════════════════════
# Display Constants
# ══════════════════════════════════════════════════════════════════════════════

HEADER_WIDTH = 60
STAGE_GAP = 4


# ══════════════════════════════════════════════════════════════════════════════
# Formatting Utilities
# ══════════════════════════════════════════════════════════════════════════════


def fmt_prob(p: float, width: int = 10) -> str:
    """Format probability, using scientific notation for very small values."""
    if p < 0.0001:
        return f"{p:>{width}.1e}"
    return f"{p:>{width}.4f}"


def fmt_core(core: list[float]) -> str:
    """Format core vector for display (full, no truncation)."""
    if not core:
        return "[]"
    items = ", ".join(f"{c:.3f}" for c in core)
    return f"[{items}]"


def oneline(text: str) -> str:
    """Collapse whitespace to single spaces for display."""
    return re.sub(r"\s+", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Logging Utilities
# ══════════════════════════════════════════════════════════════════════════════


def log_box(
    title: str,
    char: str = "═",
    subtitle: str | None = None,
    gap: int = 0,
) -> None:
    """Log a boxed header.

    Args:
        title: Main title text
        char: Border character (═ for sections, █ for major, ▓ for stages)
        subtitle: Optional second line
        gap: Lines to skip before
    """
    log(char * HEADER_WIDTH, gap=gap)
    log(f"{char}  {title}" if char in "█▓" else title)
    if subtitle:
        log(f"{char}  {subtitle}" if char in "█▓" else subtitle)
    log(char * HEADER_WIDTH)


def log_header(title: str, gap: int = 0) -> None:
    """Log a section header with double-line border."""
    log_box(title, char="═", gap=gap)


def log_major(title: str, subtitle: str | None = None, gap: int = 0) -> None:
    """Log a major section header with solid block border."""
    log_box(title, char="█", subtitle=subtitle, gap=gap)


def log_stage(step: int, total: int, title: str) -> None:
    """Log a pipeline stage separator."""
    log_box(f"STAGE {step}/{total}: {title}", char="▓", gap=STAGE_GAP)


def log_step(step_num: int, title: str, detail: str = "") -> None:
    """Log a step header with consistent formatting."""
    header = f"  Step {step_num}: {title}"
    if detail:
        header += f" ({detail})"
    log(f"\n{header}")
    log("  " + "─" * 50)


def log_divider(width: int = 62, indent: str = "  ") -> None:
    """Log a horizontal divider line."""
    log(indent + "─" * width)


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
            parts.append(f"{label:<{width}}")
        elif align == ">":
            parts.append(f"{label:>{width}}")
        else:
            parts.append(f"{label:^{width}}")
    log(indent + "  ".join(parts))
    log_divider(divider_width, indent)


def log_section_title(title: str, indent: str = "  ") -> None:
    """Log a section title within a display block."""
    log("")
    log(f"{indent}{title}")


def log_banner(title: str, char: str = "═", width: int = 70) -> None:
    """Log a banner header for summarize() methods.

    Args:
        title: Title text
        char: Border character (═ for major sections, ─ for sub-sections)
        width: Total width of the banner
    """
    log("\n" + char * width)
    log(title)
    log(char * width)


def log_sub_banner(title: str, width: int = 70) -> None:
    """Log a sub-section banner with single lines."""
    log_banner(title, char="─", width=width)


def log_wrapped(text: str, indent: str = "  ", width: int = 78, gap: int = 0) -> None:
    """Log text with word wrapping."""
    words = text.split()
    line = indent
    first = True
    for word in words:
        if len(line) + len(word) + 1 > width:
            log(line, gap=gap if first else 0)
            first = False
            line = indent + word
        else:
            line = line + " " + word if line != indent else indent + word
    if line.strip():
        log(line, gap=gap if first else 0)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Header Utilities
# ══════════════════════════════════════════════════════════════════════════════


def log_kv(key: str, value: str, indent: str = "  ") -> None:
    """Log a key-value pair."""
    log(f"{indent}{key}: {value}")


def log_pipeline_header(
    title: str,
    fields: dict[str, str | None],
    indent: str = "  ",
) -> None:
    """Log a pipeline header with title and key-value fields.

    Args:
        title: Banner title
        fields: Dict of label -> value (None values are skipped)
        indent: Indentation for fields
    """
    log_banner(title)
    log("")
    for key, value in fields.items():
        if value is not None:
            log_kv(key, value, indent)


def log_items(
    header: str,
    items: list[str | list[str]],
    prefix: str = "",
    indent: str = "    ",
) -> None:
    """Log a list of items with optional bundling.

    Args:
        header: Section header (e.g., "Categorical judgments (3):")
        items: List of strings or bundled lists
        prefix: Label prefix (e.g., "c" for c1, c2, ...)
        indent: Indentation
    """
    log(f"  {header}")
    for i, item in enumerate(items):
        label = f"[{prefix}{i+1}]" if prefix else f"[{i+1}]"
        if isinstance(item, list):
            log(f"{indent}{label} BUNDLED ({len(item)} items):")
            for sub in item:
                log(f"{indent}  • {sub}")
        else:
            log(f"{indent}{label} {item}")
