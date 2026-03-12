"""Header and banner formatting for pipeline output.

Provides consistent section headers, banners, and dividers using Unicode box-drawing chars.
"""

from __future__ import annotations

from .log_primitives import log

# Display constants
HEADER_WIDTH = 60
BANNER_WIDTH = 70
STAGE_GAP = 4


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


def log_divider(width: int = 62, indent_str: str = "  ") -> None:
    """Log a horizontal divider line."""
    log(indent_str + "─" * width)


def log_banner(title: str, char: str = "═", width: int = BANNER_WIDTH) -> None:
    """Log a banner header for summarize() methods.

    Args:
        title: Title text
        char: Border character (═ for major sections, ─ for sub-sections)
        width: Total width of the banner
    """
    log("\n" + char * width)
    log(title)
    log(char * width)


def log_sub_banner(title: str, width: int = BANNER_WIDTH) -> None:
    """Log a sub-section banner with single lines."""
    log_banner(title, char="─", width=width)


def log_section_title(title: str, indent_str: str = "  ") -> None:
    """Log a section title within a display block."""
    log("")
    log(f"{indent_str}{title}")


def log_pipeline_header(
    title: str,
    fields: dict[str, str | None],
    indent_str: str = "  ",
) -> None:
    """Log a pipeline header with title and key-value fields.

    Args:
        title: Banner title
        fields: Dict of label -> value (None values are skipped)
        indent_str: Indentation for fields
    """
    log_banner(title)
    log("")
    for key, value in fields.items():
        if value is not None:
            log(f"{indent_str}{key}: {value}")
