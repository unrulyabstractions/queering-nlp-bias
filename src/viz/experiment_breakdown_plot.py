"""Structure breakdown visualization.

Shows all structures (both categorical and bundled) with per-branch percentages
as grouped horizontal bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.estimation.arm_types import get_arm_color

from .viz_plot_utils import save_figure, style_axis_clean


def plot_structure_breakdown(
    arm_scoring: list[dict[str, Any]],
    structure_info: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    """Create grouped horizontal bar chart showing question breakdown per branch.

    Shows ALL structures:
    - Categorical (non-bundled): single questions like c1, c2
    - Bundled: grouped questions like g1, g2

    Args:
        arm_scoring: List of per-branch scoring dicts with simple_scoring and bundled_scoring
        structure_info: List of structure info dicts with is_bundled, questions, description
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no structures
    """
    if not structure_info:
        return None

    # Get arm names
    arm_names = [arm["arm"] for arm in arm_scoring]
    n_arms = len(arm_names)

    # Collect all questions grouped by structure
    # Each entry: (label, [(question_text, is_simple)])
    struct_questions: list[tuple[str, list[tuple[str, bool]]]] = []

    for struct in structure_info:
        label = struct["label"]
        is_bundled = struct.get("is_bundled", False)

        if is_bundled:
            # Bundled structure: multiple questions
            questions = struct.get("questions", [])
            if questions:
                struct_questions.append((label, [(q, False) for q in questions]))
        else:
            # Simple/categorical structure: single question
            description = struct.get("description", label)
            struct_questions.append((label, [(description, True)]))

    if not struct_questions:
        return None

    # Count total questions
    total_questions = sum(len(qs) for _, qs in struct_questions)
    n_structs = len(struct_questions)

    # Create figure with space for structure labels on left
    fig_height = max(4, total_questions * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(12, fig_height))

    # Build data for plotting with gaps between structures
    y_positions = []
    structure_spans = []  # (start_pos, end_pos, label)
    question_is_simple = []  # Track which questions are simple
    current_pos = 0
    gap = 0.8  # Gap between structure groups

    for struct_idx, (label, questions) in enumerate(struct_questions):
        if struct_idx > 0:
            current_pos += gap

        start_pos = current_pos
        for q_text, is_simple in questions:
            y_positions.append(current_pos)
            question_is_simple.append(is_simple)
            current_pos += 1
        end_pos = current_pos - 1
        structure_spans.append((start_pos, end_pos, label))

    y_positions = np.array(y_positions)
    bar_height = 0.75 / n_arms

    # Plot bars for each branch
    for branch_idx, arm in enumerate(arm_scoring):
        bundled = arm.get("bundled_scoring", {})
        simple = arm.get("simple_scoring", {})

        values = []
        q_idx = 0
        for label, questions in struct_questions:
            for q_text, is_simple in questions:
                if is_simple:
                    # Simple/categorical: get from simple_scoring
                    val = simple.get(label, 0.0)
                else:
                    # Bundled: get from bundled_scoring
                    bundle = bundled.get(label, {})
                    items = bundle.get("items", {})
                    val = items.get(q_text, items.get(q_text[:50], 0.0))
                values.append(val * 100)
                q_idx += 1

        offset = (branch_idx - n_arms / 2 + 0.5) * bar_height
        color = get_arm_color(arm["arm"])

        bars = ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            label=arm["arm"],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

        # Add value labels (only when bars are wide enough)
        for bar, val in zip(bars, values):
            if val > 15:
                ax.text(
                    bar.get_width() - 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="right", va="center",
                    color="white", fontsize=7, fontweight="bold"
                )

    # Build y-tick labels (questions only)
    y_labels = []
    for _, questions in struct_questions:
        for q_text, _ in questions:
            q_short = q_text[:45] + "..." if len(q_text) > 48 else q_text
            y_labels.append(q_short)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)

    # Add alternating background bands and structure labels
    band_colors = ["#f8f8f8", "#ffffff"]
    for i, (start_pos, end_pos, label) in enumerate(structure_spans):
        mid_pos = (start_pos + end_pos) / 2

        # Background band
        ax.axhspan(
            start_pos - 0.45, end_pos + 0.45,
            color=band_colors[i % 2],
            alpha=0.6, zorder=0
        )

        # Vertical line on left edge (neutral blue accent)
        ax.plot(
            [0, 0], [start_pos - 0.45, end_pos + 0.45],
            color="#4A90D9", linewidth=3, alpha=0.7,
            transform=ax.get_yaxis_transform(), clip_on=False
        )

        # Structure label - small, positioned at top-left of each group
        ax.text(
            -0.02, start_pos - 0.35,
            label.lower(),
            ha="right", va="bottom",
            fontsize=8, fontweight="bold",
            color="#666",
            transform=ax.get_yaxis_transform(),
        )

    # Set x-axis
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])

    # Title
    ax.set_title(
        "Structure Breakdown by Branch",
        fontsize=13, fontweight="bold", pad=15
    )

    # Grid and styling
    style_axis_clean(ax, grid_axis="x", remove_top_spine=True, remove_right_spine=True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Invert y-axis
    ax.invert_yaxis()

    # Legend at bottom - very compact (2 rows max)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=min(n_arms, 4),
        fontsize=7,
        framealpha=0.9,
        handlelength=0.8,
        handletextpad=0.3,
        columnspacing=0.6,
    )

    # Save
    plt.subplots_adjust(left=0.30)
    save_figure(plt.gcf(), output_path)
    return output_path
