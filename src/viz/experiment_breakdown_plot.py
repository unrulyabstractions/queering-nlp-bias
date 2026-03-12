"""Bundled structure breakdown visualization.

Shows individual questions within bundled structures with per-branch percentages
as grouped horizontal bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Branch colors (consistent with other plots)
BRANCH_COLORS = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C"]


def plot_bundled_structures(
    arm_scoring: list[dict[str, Any]],
    structure_info: list[dict[str, Any]],
    output_path: Path,
) -> Path | None:
    """Create grouped horizontal bar chart showing question breakdown per branch.

    Only shows bundled structures (those with multiple questions).

    Args:
        arm_scoring: List of per-branch scoring dicts with bundled_scoring
        structure_info: List of structure info dicts with is_bundled, questions
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no bundled structures
    """
    # Filter to bundled structures only
    bundled_structs = [s for s in structure_info if s.get("is_bundled", False)]
    if not bundled_structs:
        return None

    # Get branch names
    branch_names = [arm["branch"] for arm in arm_scoring]
    n_branches = len(branch_names)

    # Collect all questions grouped by structure
    struct_questions: list[tuple[str, list[str]]] = []  # [(label, [questions])]
    for struct in bundled_structs:
        label = struct["label"]
        questions = struct.get("questions", [])
        if questions:
            struct_questions.append((label, questions))

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
    current_pos = 0
    gap = 0.8  # Gap between structure groups

    for struct_idx, (label, questions) in enumerate(struct_questions):
        if struct_idx > 0:
            current_pos += gap

        start_pos = current_pos
        for _ in questions:
            y_positions.append(current_pos)
            current_pos += 1
        end_pos = current_pos - 1
        structure_spans.append((start_pos, end_pos, label))

    y_positions = np.array(y_positions)
    bar_height = 0.75 / n_branches

    # Plot bars for each branch
    for branch_idx, arm in enumerate(arm_scoring):
        bundled = arm.get("bundled_scoring", {})

        values = []
        for label, questions in struct_questions:
            bundle = bundled.get(label, {})
            items = bundle.get("items", {})
            for q in questions:
                val = items.get(q, items.get(q[:50], 0.0))
                values.append(val * 100)

        offset = (branch_idx - n_branches / 2 + 0.5) * bar_height
        color = BRANCH_COLORS[branch_idx % len(BRANCH_COLORS)]

        bars = ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            label=arm["branch"],
            color=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 8:
                ax.text(
                    bar.get_width() - 2, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="right", va="center",
                    color="white", fontsize=8, fontweight="bold"
                )

    # Build y-tick labels (questions only)
    y_labels = []
    for _, questions in struct_questions:
        for q in questions:
            q_short = q[:45] + "..." if len(q) > 48 else q
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

        # Vertical line on left edge
        ax.plot(
            [0, 0], [start_pos - 0.45, end_pos + 0.45],
            color=BRANCH_COLORS[0], linewidth=4, alpha=0.8,
            transform=ax.get_yaxis_transform(), clip_on=False
        )

        # Structure label in left margin
        ax.text(
            -0.18, mid_pos,
            label.upper(),
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="#333",
            transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#999", linewidth=1.5)
        )

    # Set x-axis
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percentage (%)", fontsize=11)
    ax.set_xticks([0, 25, 50, 75, 100])

    # Title
    ax.set_title(
        "Bundled Structure Breakdown by Branch",
        fontsize=13, fontweight="bold", pad=15
    )

    # Grid
    ax.grid(axis="x", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Invert y-axis
    ax.invert_yaxis()

    # Legend at bottom
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(n_branches, 5),
        fontsize=10,
        framealpha=0.9,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.30)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path
