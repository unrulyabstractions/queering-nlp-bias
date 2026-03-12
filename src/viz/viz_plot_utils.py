"""Shared visualization utilities for matplotlib plots.

Contains common patterns extracted from visualization modules:
- Axis styling (grid, spines, limits)
- Bar value annotations
- Figure save/close patterns
- Shared color palettes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

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
) -> None:
    """Add value labels on top of bars.

    Args:
        ax: Matplotlib axis
        bars: Bar container from ax.bar() or ax.barh()
        values: Numeric values to display
        fontsize: Font size for labels
        color: Text color
        fontweight: Font weight
        format_str: Format string for values (e.g., "{:.2f}", "{:+.2f}")
        offset_points: (x, y) offset in points
        signed: If True, always show sign (+/-)
    """
    actual_format = "{:+.2f}" if signed else format_str

    for bar, val in zip(bars, values):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_offset = offset_points[1] if height >= 0 else -offset_points[1]

        ax.annotate(
            actual_format.format(val),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(offset_points[0], y_offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=fontsize,
            color=color,
            fontweight=fontweight,
        )


def save_figure(
    fig: "Figure",
    output_path: Path,
    *,
    dpi: int = 150,
    tight_layout_rect: list[float] | None = None,
    facecolor: str = "white",
) -> Path:
    """Save figure and close it, creating parent directories as needed.

    Args:
        fig: Matplotlib figure to save
        output_path: Where to save the PNG
        dpi: Output resolution
        tight_layout_rect: Optional rect for tight_layout [left, bottom, right, top]
        facecolor: Background color

    Returns:
        The output path (for chaining)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if tight_layout_rect:
        plt.tight_layout(rect=tight_layout_rect)
    else:
        plt.tight_layout()

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
