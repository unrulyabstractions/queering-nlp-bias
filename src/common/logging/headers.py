"""Header and banner formatting for pipeline output.

Provides consistent section headers, banners, and dividers.
"""

from __future__ import annotations

from .core import log
from .formatting import center

# Display constants
HEADER_WIDTH = 76
STAGE_GAP = 4


def log_box(
    title: str,
    char: str = "=",
    subtitle: str | None = None,
    width: int = HEADER_WIDTH,
    gap: int = 0,
    centered: bool = True,
) -> None:
    """Log a boxed header.

    Args:
        title: Main title text
        char: Border character (= for sections, # for major, - for sub)
        subtitle: Optional second line
        width: Box width
        gap: Lines to skip before
        centered: Whether to center the title text
    """
    border = char * width
    log(border, gap=gap)
    if centered:
        log(center(title, width))
    else:
        log(f"  {title}")
    if subtitle:
        if centered:
            log(center(subtitle, width))
        else:
            log(f"  {subtitle}")
    log(border)


def log_header(title: str, width: int = HEADER_WIDTH, gap: int = 0) -> None:
    """Log a section header with double-line border."""
    log_box(title, char="=", width=width, gap=gap, centered=True)


def log_major(
    title: str,
    subtitle: str | None = None,
    width: int = HEADER_WIDTH,
    gap: int = 0,
) -> None:
    """Log a major section header with block border."""
    log_box(title, char="#", subtitle=subtitle, width=width, gap=gap, centered=True)


def log_stage(step: int, total: int, title: str, width: int = HEADER_WIDTH) -> None:
    """Log a pipeline stage separator."""
    log_box(f"STAGE {step}/{total}: {title}", char="-", width=width, gap=STAGE_GAP)


def log_step(step_num: int, title: str, detail: str = "") -> None:
    """Log a step header with consistent formatting."""
    header = f"  Step {step_num}: {title}"
    if detail:
        header += f" ({detail})"
    log(f"\n{header}")
    log("  " + "-" * 50)


def log_divider(width: int = 62, indent: str = "  ") -> None:
    """Log a horizontal divider line."""
    log(indent + "-" * width)


def log_banner(title: str, char: str = "=", width: int = HEADER_WIDTH) -> None:
    """Log a banner header for summarize() methods.

    Args:
        title: Title text (will be centered)
        char: Border character (= for major sections, - for sub-sections)
        width: Total width of the banner
    """
    border = char * width
    log("")
    log(border)
    log(center(title, width))
    log(border)


def log_sub_banner(title: str, width: int = HEADER_WIDTH) -> None:
    """Log a sub-section banner with single lines."""
    log_banner(title, char="-", width=width)


def log_pipeline_header(
    title: str,
    fields: dict[str, str | None],
    width: int = HEADER_WIDTH,
    indent_str: str = "  ",
) -> None:
    """Log a pipeline header with title and key-value fields.

    Args:
        title: Banner title (will be centered)
        fields: Dict of label -> value (None values are skipped)
        width: Banner width
        indent_str: Indentation for fields
    """
    log_banner(title, width=width)
    log("")
    for key, value in fields.items():
        if value is not None:
            log(f"{indent_str}{key}: {value}")
