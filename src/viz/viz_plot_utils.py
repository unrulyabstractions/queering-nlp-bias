"""Shared visualization utilities for matplotlib plots.

Contains common patterns extracted from visualization modules:
- Axis styling (grid, spines, limits)
- Bar value annotations
- Figure save/close patterns
- Shared color palettes
- Camera-ready mode for publication quality
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.common.profiler import P

# Suppress matplotlib font fallback warnings (fonts work fine with fallbacks)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Global visualization settings
_VIZ_CONFIG = {
    "camera_ready": False,  # High DPI, all features enabled
    "dpi": 150,             # Default DPI (fast)
    "dpi_camera_ready": 300,  # DPI for camera-ready mode
}


def set_camera_ready(enabled: bool = True) -> None:
    """Enable or disable camera-ready mode for publication-quality plots."""
    _VIZ_CONFIG["camera_ready"] = enabled


def is_camera_ready() -> bool:
    """Check if camera-ready mode is enabled."""
    return _VIZ_CONFIG["camera_ready"]


def get_dpi() -> int:
    """Get current DPI setting based on mode."""
    if _VIZ_CONFIG["camera_ready"]:
        return _VIZ_CONFIG["dpi_camera_ready"]
    return _VIZ_CONFIG["dpi"]

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.figure import Figure


# Shared color palette for structures - consistent across all visualizations
STRUCTURE_COLORS = [
    "#4A90D9",  # blue
    "#E67E22",  # orange
    "#2ECC71",  # green
    "#E74C3C",  # red
    "#9B59B6",  # purple
    "#1ABC9C",  # teal
    "#F39C12",  # yellow
    "#E91E63",  # pink
    "#5DA5DA",  # light blue
    "#B276B2",  # light purple
]


def get_structure_color(idx: int) -> str:
    """Get color for a structure by index, cycling through the palette."""
    return STRUCTURE_COLORS[idx % len(STRUCTURE_COLORS)]


def style_axis_clean(
    ax: "Axes",
    *,
    remove_top_spine: bool = True,
    remove_right_spine: bool = True,
    grid_axis: str | None = "y",
    grid_alpha: float = 0.3,
    grid_linestyle: str = ":",
) -> None:
    """Apply clean styling to an axis.

    Removes spines, adds subtle grid, and applies consistent styling.

    Args:
        ax: Matplotlib axis to style
        remove_top_spine: Whether to remove top spine
        remove_right_spine: Whether to remove right spine
        grid_axis: Which axis to add grid to ("x", "y", "both", or None)
        grid_alpha: Grid transparency
        grid_linestyle: Grid line style
    """
    if remove_top_spine:
        ax.spines["top"].set_visible(False)
    if remove_right_spine:
        ax.spines["right"].set_visible(False)

    if grid_axis:
        ax.grid(axis=grid_axis, alpha=grid_alpha, linestyle=grid_linestyle)


def annotate_bar_values(
    ax: "Axes",
    bars: "BarContainer",
    values: list[float],
    *,
    fontsize: int = 8,
    color: str = "#333333",
    fontweight: str = "medium",
    format_str: str = "{:.2f}",
    offset_points: tuple[int, int] = (0, 3),
    signed: bool = False,
    n_bars_per_group: int | None = None,
) -> None:
    """Add value labels on top of bars.

    Args:
        ax: Matplotlib axis
        bars: Bar container from ax.bar() or ax.barh()
        values: Numeric values to display
        fontsize: Font size for labels (will be scaled down if many bars)
        color: Text color
        fontweight: Font weight
        format_str: Format string for values (e.g., "{:.2f}", "{:+.2f}")
        offset_points: (x, y) offset in points
        signed: If True, always show sign (+/-)
        n_bars_per_group: Number of bars per group (for dynamic font scaling)
    """
    # Dynamically scale font size based on number of bars
    if n_bars_per_group is not None and n_bars_per_group > 4:
        # Scale down more aggressively: 5 bars -> 0.75x, 10 bars -> 0.38x
        scale = max(0.35, 1.0 - (n_bars_per_group - 4) * 0.1)
        fontsize = max(4, int(fontsize * scale))
        # Use fewer decimals for very small fonts
        if fontsize <= 5 and format_str == "{:.2f}":
            format_str = "{:.1f}"

    actual_format = "{:+.2f}" if signed else format_str

    # Rotate labels if very crowded
    rotation = 45 if n_bars_per_group is not None and n_bars_per_group >= 8 else 0
    ha = "left" if rotation else "center"

    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_offset = offset_points[1] if height >= 0 else -offset_points[1]

        ax.annotate(
            actual_format.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(offset_points[0], y_offset),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
            rotation=rotation,
        )


def save_figure(
    fig: "Figure",
    output_path: Path,
    *,
    dpi: int | None = None,  # None = use global setting
    tight_layout_rect: list[float] | None = None,
    facecolor: str = "white",
    skip_tight_layout: bool = False,
) -> Path:
    """Save figure and close it, creating parent directories as needed.

    Args:
        fig: Matplotlib figure to save
        output_path: Where to save the PNG
        dpi: Output resolution (None = use global get_dpi())
        tight_layout_rect: Optional rect for tight_layout [left, bottom, right, top]
        facecolor: Background color
        skip_tight_layout: If True, skip tight_layout call (for manual layout)

    Returns:
        The output path (for chaining)
    """
    if dpi is None:
        dpi = get_dpi()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not skip_tight_layout:
        with P("tight_layout"):
            if tight_layout_rect:
                plt.tight_layout(rect=tight_layout_rect)
            else:
                plt.tight_layout()

    with P("savefig"):
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=facecolor)

    plt.close()

    return output_path


def add_reference_line(
    ax: "Axes",
    y: float,
    *,
    color: str = "#cccccc",
    linestyle: str = "--",
    linewidth: float = 0.8,
    label: str | None = None,
    zorder: int = 0,
) -> None:
    """Add a horizontal reference line to an axis.

    Args:
        ax: Matplotlib axis
        y: Y-value for the line
        color: Line color
        linestyle: Line style
        linewidth: Line width
        label: Optional label for legend
        zorder: Drawing order
    """
    ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth,
               zorder=zorder, label=label)


def lighten_color(hex_color: str, factor: float = 0.5) -> str:
    """Lighten a hex color by blending with white.

    Args:
        hex_color: Color in hex format (e.g., "#4A90D9")
        factor: Blend factor (0 = original, 1 = white)

    Returns:
        Lightened hex color
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)

    return f"#{r:02x}{g:02x}{b:02x}"


def create_arm_legend(
    ax: "Axes",
    arm_names: list[str],
    arm_descriptions: dict[str, str] | None = None,
    *,
    max_desc_length: int = 35,
    fontsize: int = 9,
    bbox_anchor: tuple[float, float] = (1.02, 1),
    loc: str = "upper left",
    include_spacing: bool = True,
) -> None:
    """Create a styled hierarchical legend with arm names and descriptions.

    Typography (as requested):
    - Arm names: Roboto Mono (or monospace fallback), bold, slightly gray
    - Descriptions: Akinson Hyperlegible (or sans-serif fallback), regular, black

    Hierarchy shown via indentation:
    - root/trunk: no indent
    - branches: 1 tab indent
    - twigs: 2 tabs indent

    Args:
        ax: Matplotlib axis to add legend to
        arm_names: List of arm names in order
        arm_descriptions: Optional dict mapping arm names to conditioning text
        max_desc_length: Maximum description length before truncation
        fontsize: Legend font size
        bbox_anchor: Legend position (x, y) relative to axes
        loc: Legend location anchor point
        include_spacing: Whether to add spacing between arm families
    """
    with P("legend_imports"):
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import FancyBboxPatch, Rectangle

        from src.estimation.arm_types import (
            ArmKind,
            classify_arm,
            get_arm_color,
            get_branch_index,
        )

    with P("legend_setup"):
        # Font properties - use the actual fontsize passed in
        arm_font = FontProperties(
            family=['Roboto Mono', 'DejaVu Sans Mono', 'Consolas', 'monospace'],
            weight='bold',
            size=fontsize,  # Use actual fontsize
        )
        # Descriptions: Akinson Hyperlegible, italic
        desc_font = FontProperties(
            family=['Akinson Hyperlegible', 'DejaVu Sans', 'Helvetica', 'sans-serif'],
            weight='regular',
            style='italic',
            size=fontsize + 2,  # Descriptions slightly bigger
        )

        # Group arms by family
        family_groups: dict[int, list[str]] = {}
        for name in arm_names:
            kind = classify_arm(name)
            if kind in (ArmKind.ROOT, ArmKind.TRUNK):
                family_idx = 0
            else:
                branch_idx = get_branch_index(name)
                family_idx = branch_idx if branch_idx else 99

            if family_idx not in family_groups:
                family_groups[family_idx] = []
            family_groups[family_idx].append(name)

        # Sort within each family
        arm_family_order = []
        for family_idx in sorted(family_groups.keys()):
            members = family_groups[family_idx]
            members.sort(key=lambda n: (
                0 if classify_arm(n) == ArmKind.ROOT else
                1 if classify_arm(n) == ArmKind.TRUNK else
                2 if classify_arm(n) == ArmKind.BRANCH else 3,
                n
            ))
            for name in members:
                arm_family_order.append((name, family_idx))

        arm_descs = arm_descriptions or {}
        has_descriptions = bool(arm_descs)

    # Layout - tab-based indentation (big tabs for clear hierarchy)
    # Scale spacing proportionally to fontsize (baseline fontsize=9)
    scale = fontsize / 9.0
    x_start = bbox_anchor[0]
    y_start = bbox_anchor[1]
    tab_width = 0.045 * min(scale, 1.5)  # Tabs don't need to scale as much
    line_height = (0.070 if has_descriptions else 0.060) * scale  # More vertical spacing
    desc_line_height = 0.040 * scale  # More space for descriptions
    swatch_size = 0.032 * min(scale, 1.5)  # Swatch scales moderately
    box_width = 0.52 + 0.15 * (scale - 1.0)  # Box width grows slowly

    # Calculate total height
    total_height = 0.02
    prev_fam = None
    for _, fam in arm_family_order:
        if include_spacing and prev_fam is not None and fam != prev_fam:
            total_height += line_height * 0.5
        total_height += line_height
        prev_fam = fam
    if has_descriptions:
        total_height += desc_line_height * len(arm_family_order)
    total_height += 0.015

    # Adjust y_start for center positioning (legend draws downward from y_start)
    if "center" in loc.lower():
        # Center the legend vertically around bbox_anchor[1]
        y_start = bbox_anchor[1] + total_height / 2 - 0.02  # -0.02 for top padding

    with P("legend_draw"):
        # Draw background box (no border)
        bg_rect = FancyBboxPatch(
            (x_start - 0.01, y_start - total_height),
            box_width, total_height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            transform=ax.transAxes,
            facecolor='white',
            edgecolor='none',  # No border
            linewidth=0,
            alpha=0.95,
            zorder=100,
            clip_on=False,
        )
        ax.add_patch(bg_rect)

        # Draw entries
        y_pos = y_start - 0.02
        prev_family = None

        for arm_name, family_idx in arm_family_order:
            color = get_arm_color(arm_name)
            kind = classify_arm(arm_name)

            # Add spacing between families (visual grouping)
            if include_spacing and prev_family is not None and family_idx != prev_family:
                y_pos -= line_height * 0.5

            # Tab-based indent for hierarchy:
            # root = 0 tabs, trunk = 1 tab, branch = 2 tabs, twig = 3 tabs
            if kind == ArmKind.ROOT:
                indent = 0.0
            elif kind == ArmKind.TRUNK:
                indent = tab_width * 1
            elif kind == ArmKind.BRANCH:
                indent = tab_width * 2
            else:  # TWIG
                indent = tab_width * 3

            # Draw color swatch
            swatch = Rectangle(
                (x_start + indent, y_pos - swatch_size / 2),
                swatch_size, swatch_size,
                transform=ax.transAxes,
                facecolor=color,
                edgecolor='white',
                linewidth=0.5,
                zorder=101,
                clip_on=False,
            )
            ax.add_patch(swatch)

            # Draw arm name: Roboto Mono, bold, slightly gray
            text_x = x_start + indent + swatch_size + 0.008
            ax.text(
                text_x, y_pos,
                arm_name,
                transform=ax.transAxes,
                fontproperties=arm_font,
                color='#666',  # Slightly gray
                va='center',
                ha='left',
                zorder=101,
                clip_on=False,
            )

            y_pos -= line_height

            # Draw description on second line: Akinson Hyperlegible, regular, black
            desc = arm_descs.get(arm_name, "")
            if desc:
                desc_short = desc[:max_desc_length] + "..." if len(desc) > max_desc_length else desc
                ax.text(
                    text_x + 0.008, y_pos + 0.01,
                    desc_short,
                    transform=ax.transAxes,
                    fontproperties=desc_font,
                    color='#222',  # Black
                    va='center',
                    ha='left',
                    zorder=101,
                    clip_on=False,
                )
                y_pos -= desc_line_height

            prev_family = family_idx
