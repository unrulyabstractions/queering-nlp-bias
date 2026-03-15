"""Matplotlib rendering functions for forking tree visualizations.

Contains all matplotlib-specific drawing code:
- Legend rendering
- Tree node rendering
- Connecting lines
- Reference lines
- Metadata display
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from .viz_bounding_box import TreeContentTracker
from .viz_plot_utils import get_structure_color


def draw_wrapped_arm_label(
    ax: Any,
    x: float,
    y: float,
    text: str,
    fontsize: float,
    max_width: float,
    *,
    color: str = "#000",
    fontweight: str = "bold",
) -> None:
    """Draw arm label text, wrapping to multiple left-aligned lines if too wide.

    Args:
        ax: Matplotlib axes
        x: X position (left edge)
        y: Y position (bottom of first line)
        text: Text to display
        fontsize: Font size in points
        max_width: Maximum width in data units before wrapping
        color: Text color
        fontweight: Font weight
    """
    if not text:
        return

    # Estimate character width in data units (rough approximation)
    # At fontsize 16, average char is about 0.12 data units wide
    char_width = fontsize * 0.008

    # Estimate text width
    estimated_width = len(text) * char_width

    if estimated_width <= max_width:
        # Single line - no wrapping needed
        ax.text(
            x, y, text,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha="left",
            va="bottom",
            zorder=10,
        )
        return

    # Need to wrap - split into words and build lines
    words = text.split()
    lines = []
    current_line = []
    current_width = 0.0

    for word in words:
        word_width = len(word) * char_width
        space_width = char_width

        test_width = current_width + (space_width if current_line else 0) + word_width

        if test_width <= max_width or not current_line:
            current_line.append(word)
            current_width = test_width
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = word_width

    if current_line:
        lines.append(" ".join(current_line))

    # Draw each line, stacking upward from the base position
    line_spacing = fontsize * 0.018  # Vertical spacing between lines
    for i, line in enumerate(reversed(lines)):
        line_y = y + i * line_spacing
        ax.text(
            x, line_y, line,
            fontsize=fontsize,
            fontweight=fontweight,
            color=color,
            ha="left",
            va="bottom",
            zorder=10,
        )


def render_legend(
    ax: Any,
    layout: dict[str, Any],
    legend_top_y: float,
) -> None:
    """Render legend from precomputed layout.

    Args:
        ax: Matplotlib axes
        layout: Layout from compute_legend_layout
        legend_top_y: Y coordinate for top of legend in data units
    """
    swatch_size = layout["swatch_size"]
    char_width = layout.get("char_width", 0.055)

    # Scale fontsize based on char_width (base: 0.055 -> 9pt)
    fontsize = int(9 * (char_width / 0.055))

    # TIGHT line spacing for wrapped text - based on char_width, not row_height
    line_height = char_width * 2.0

    for item in layout["items"]:
        idx = item["index"]
        lines = item.get("lines", [item["description"]])
        n_lines = item.get("n_lines", 1)
        color = get_structure_color(idx)

        # Layout positions are relative to legend top (negative Y)
        # Add legend_top_y to get absolute position
        swatch_x = item["swatch_x"]
        swatch_y = legend_top_y + item["swatch_y"]
        text_x = item["text_x"]
        text_y = legend_top_y + item["text_y"]

        # Draw swatch (centered vertically with text block)
        ax.add_patch(
            plt.Rectangle(
                (swatch_x, swatch_y - swatch_size / 2),
                swatch_size,
                swatch_size,
                facecolor=color,
                edgecolor="white",
                linewidth=0.8,
                clip_on=False,
                zorder=100,
            )
        )

        # Draw text lines (stacked vertically)
        if n_lines == 1:
            # Single line - center aligned
            ax.text(
                text_x, text_y, lines[0],
                fontsize=fontsize, fontweight="bold",
                va="center", ha="left", color="#111", zorder=100,
            )
        else:
            # Multiple lines - start from top of text area
            text_top_y = text_y + (n_lines - 1) * line_height / 2
            for line_i, line_text in enumerate(lines):
                line_y = text_top_y - line_i * line_height
                ax.text(
                    text_x, line_y, line_text,
                    fontsize=fontsize, fontweight="bold",
                    va="center", ha="left", color="#111", zorder=100,
                )


def draw_metadata(
    fig: Any,
    metadata: dict[str, str] | None,
    n_structures: int,
) -> None:
    """Draw model and judge metadata on the figure.

    Uses CONSISTENT sizing and spacing across all plots.
    """
    if not metadata:
        return

    # FIXED font sizes for consistency across all plots
    LABEL_FONTSIZE = 14
    VALUE_FONTSIZE = 18
    LABEL_X = 0.015
    VALUE_X = 0.095  # Consistent alignment for values

    # Consistent vertical positions
    JUDGE_Y = 0.045
    MODEL_Y = 0.015

    if metadata.get("judge"):
        fig.text(
            LABEL_X, JUDGE_Y, "Judge LLM:",
            fontsize=LABEL_FONTSIZE,
            fontfamily='monospace',
            va="bottom", ha="left", color="#666",
        )
        fig.text(
            VALUE_X, JUDGE_Y, metadata['judge'],
            fontsize=VALUE_FONTSIZE,
            fontweight='bold',
            va="bottom", ha="left", color="#222",
        )

    if metadata.get("model"):
        fig.text(
            LABEL_X, MODEL_Y, "Gen Model:",
            fontsize=LABEL_FONTSIZE,
            fontfamily='monospace',
            va="bottom", ha="left", color="#666",
        )
        fig.text(
            VALUE_X, MODEL_Y, metadata['model'],
            fontsize=VALUE_FONTSIZE,
            fontweight='bold',
            va="bottom", ha="left", color="#222",
        )


def populate_tree_content_tracker(
    positions: list[dict],
    arm_texts: dict[str, str],
    n_structures: int,
    bar_height: float,
    bar_width_scale: float,
    arm_label_fontsize: float,
    x_spacing: float,
) -> TreeContentTracker:
    """Populate TreeContentTracker from computed tree positions.

    This estimates all bounding boxes for tree content without rendering,
    allowing the legend optimizer to find collision-free placements.
    """
    tracker = TreeContentTracker()

    for pos in positions:
        arm_name = pos["name"]
        x = pos["x"]
        y = pos["y"]

        # Node box bounds
        box_height = n_structures * bar_height + 0.22
        box_width = bar_width_scale + 0.25
        tracker.add_node_box(x - 0.1, y, box_width, box_height + 0.2)

        # Node label bounds (text above the box)
        raw_text = arm_texts.get(arm_name, arm_name)
        label_y = y + n_structures * bar_height / 2 + 0.3
        tracker.add_node_label(x - 0.05, label_y, raw_text, arm_label_fontsize)

        # Arm name label below box
        arm_label_y = y - n_structures * bar_height / 2 - 0.4
        tracker.add_node_label(
            x + bar_width_scale / 2 - len(arm_name) * 0.03,
            arm_label_y - 0.3,
            arm_name,
            arm_label_fontsize - 4,
        )

        # Edges to children
        if "child_ys" in pos:
            parent_x = x + bar_width_scale + 0.1
            for child_y in pos["child_ys"]:
                child_x = x + x_spacing
                tracker.add_edge(parent_x, y, child_x - 0.3, child_y, line_width=0.15)

    return tracker


def draw_connecting_lines(
    ax: Any,
    positions: list[dict],
    arm_normalized_probs: dict[str, float],
    bar_width_scale: float,
    x_spacing: float,
) -> None:
    """Draw probability-proportional connecting lines between tree nodes."""
    pos_by_y: dict[float, dict] = {round(pos["y"], 6): pos for pos in positions}

    def find_pos_by_y(target_y: float) -> dict | None:
        rounded = round(target_y, 6)
        if rounded in pos_by_y:
            return pos_by_y[rounded]
        for y, pos in pos_by_y.items():
            if abs(y - target_y) < 0.001:
                return pos
        return None

    for pos in positions:
        if "child_ys" not in pos:
            continue

        parent_x = pos["x"] + bar_width_scale + 0.1
        parent_y = pos["y"]

        for child_y in pos["child_ys"]:
            child_pos = find_pos_by_y(child_y)
            if not child_pos:
                continue

            child_name = child_pos["name"]
            norm_prob = arm_normalized_probs.get(child_name, 0.5)

            # Line thickness and color proportional to probability
            line_width = 1.0 + norm_prob * 10.0
            gray_val = int(180 - norm_prob * 140)
            line_color = f"#{gray_val:02x}{gray_val:02x}{gray_val:02x}"

            # Draw curved connector
            child_x = pos["x"] + x_spacing
            mid_x = (parent_x + child_x) / 2
            ax.plot(
                [parent_x, mid_x, mid_x, child_x - 0.3],
                [parent_y, parent_y, child_y, child_y],
                color=line_color,
                linewidth=line_width,
                solid_capstyle="round",
                zorder=1,
            )


def draw_reference_lines_core(
    ax: Any,
    x: float,
    y: float,
    box_height: float,
    bar_width_scale: float,
) -> None:
    """Draw reference lines at 0%, 50%, 100% for core/forking plots."""
    ref_y_min = y - box_height / 2
    ref_y_max = y + box_height / 2

    # 0% - RED SOLID (matches orientation 0% for visual consistency)
    ax.plot(
        [x, x],
        [ref_y_min, ref_y_max],
        color="#C0392B",
        linestyle="-",
        linewidth=3.0,
        zorder=10,
    )

    # 50% - BLACK DASHED (distinctive midpoint for forking)
    ax.plot(
        [x + 0.5 * bar_width_scale, x + 0.5 * bar_width_scale],
        [ref_y_min, ref_y_max],
        color="#000",
        linestyle="--",
        linewidth=2.5,
        zorder=10,
    )

    # 100% - thin gray dotted
    ax.plot(
        [x + bar_width_scale, x + bar_width_scale],
        [ref_y_min, ref_y_max],
        color="#999",
        linestyle=":",
        linewidth=1.0,
        zorder=8,
    )


def draw_reference_lines_orientation(
    ax: Any,
    center_x: float,
    y: float,
    box_height: float,
    bar_width_scale: float,
) -> None:
    """Draw reference lines at -100%, -50%, 0%, 50%, 100% for orientation plots."""
    ref_y_min = y - box_height / 2
    ref_y_max = y + box_height / 2
    half_width = bar_width_scale / 2

    # -100% - left edge, gray dotted
    ax.plot(
        [center_x - half_width, center_x - half_width],
        [ref_y_min, ref_y_max],
        color="#999",
        linestyle=":",
        linewidth=1.0,
        zorder=8,
    )

    # -50% - BLACK DASHED (matches forking 50% style)
    ax.plot(
        [center_x - half_width / 2, center_x - half_width / 2],
        [ref_y_min, ref_y_max],
        color="#000",
        linestyle="--",
        linewidth=2.5,
        zorder=10,
    )

    # 0% - RED SOLID (matches forking 0% style)
    ax.plot(
        [center_x, center_x],
        [ref_y_min, ref_y_max],
        color="#C0392B",
        linestyle="-",
        linewidth=3.0,
        zorder=10,
    )

    # 50% - BLACK DASHED (matches forking 50% style)
    ax.plot(
        [center_x + half_width / 2, center_x + half_width / 2],
        [ref_y_min, ref_y_max],
        color="#000",
        linestyle="--",
        linewidth=2.5,
        zorder=10,
    )

    # 100% - right edge, gray dotted
    ax.plot(
        [center_x + half_width, center_x + half_width],
        [ref_y_min, ref_y_max],
        color="#999",
        linestyle=":",
        linewidth=1.0,
        zorder=8,
    )


def desaturate_color(hex_color: str, factor: float = 0.5) -> str:
    """Desaturate a hex color by blending with gray."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    gray = (r + g + b) // 3
    r = int(r * factor + gray * (1 - factor))
    g = int(g * factor + gray * (1 - factor))
    b = int(b * factor + gray * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"
