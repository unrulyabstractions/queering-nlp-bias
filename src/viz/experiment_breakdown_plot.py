"""Structure breakdown visualization.

Shows all structures (both categorical and bundled) with per-branch percentages
as grouped horizontal bar charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.common.profiler import P
from src.estimation.arm_types import (
    classify_arm,
    get_arm_color,
    get_branch_index,
    ArmKind,
)

from .viz_plot_utils import create_arm_legend, save_figure, style_axis_clean


def _get_arm_family_order(arm_names: list[str]) -> list[tuple[str, int]]:
    """Sort arms by family and return (arm_name, family_idx) pairs.

    Groups: baseline (root, trunk), then branch_1 family, branch_2 family, etc.
    Returns list of (arm_name, family_index) for spacing.
    """
    family_groups: dict[int, list[str]] = {}

    for name in arm_names:
        kind = classify_arm(name)
        if kind in (ArmKind.ROOT, ArmKind.TRUNK):
            family_idx = 0  # Baseline family
        else:
            branch_idx = get_branch_index(name)
            family_idx = branch_idx if branch_idx else 99

        if family_idx not in family_groups:
            family_groups[family_idx] = []
        family_groups[family_idx].append(name)

    # Sort within each family: branch first, then twigs
    result = []
    for family_idx in sorted(family_groups.keys()):
        members = family_groups[family_idx]
        # Sort: root/trunk/branch before twigs
        members.sort(key=lambda n: (
            0 if classify_arm(n) == ArmKind.ROOT else
            1 if classify_arm(n) == ArmKind.TRUNK else
            2 if classify_arm(n) == ArmKind.BRANCH else 3,
            n
        ))
        for name in members:
            result.append((name, family_idx))

    return result


def plot_structure_breakdown(
    arm_scoring: list[dict[str, Any]],
    structure_info: list[dict[str, Any]],
    output_path: Path,
    arm_descriptions: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Path | None:
    """Create grouped horizontal bar chart showing question breakdown per branch.

    Shows ALL structures:
    - Categorical (non-bundled): single questions like c1, c2
    - Bundled: grouped questions like g1, g2

    Args:
        arm_scoring: List of per-branch scoring dicts with simple_scoring and bundled_scoring
        structure_info: List of structure info dicts with is_bundled, questions, description
        output_path: Where to save the plot
        arm_descriptions: Optional dict mapping arm names to their conditioning text
        metadata: Optional dict with 'prompt', 'model', 'judge' keys for display

    Returns:
        Path to saved file, or None if no structures
    """
    if not structure_info:
        return None

    with P("breakdown_data_prep"):
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

        # Get arm family ordering with family indices
        arm_family_order = _get_arm_family_order(arm_names)
        ordered_arms = [name for name, _ in arm_family_order]

        # Create arm lookup for data
        arm_data_lookup = {arm["arm"]: arm for arm in arm_scoring}

        # Calculate bar positions with gaps between families
        bar_height = 0.65 / n_arms
        family_gap = bar_height * 0.6  # Extra gap between families

        # Calculate offsets for each arm (with family gaps)
        arm_offsets = {}
        current_offset = -n_arms / 2 * bar_height
        prev_family = None

        for arm_name, family_idx in arm_family_order:
            if prev_family is not None and family_idx != prev_family:
                current_offset += family_gap  # Add gap between families
            arm_offsets[arm_name] = current_offset
            current_offset += bar_height
            prev_family = family_idx

        # Center the offsets
        total_width = current_offset + n_arms / 2 * bar_height
        center_adjustment = -total_width / 2 + bar_height / 2
        for arm_name in arm_offsets:
            arm_offsets[arm_name] += center_adjustment

    with P("breakdown_figure_create"):
        # Create figure with space for structure labels on left
        fig_height = max(4, total_questions * 0.7 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))

    with P("breakdown_draw_bars"):
        # Plot bars for each arm in family order
        for arm_name in ordered_arms:
            arm = arm_data_lookup[arm_name]
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

            offset = arm_offsets[arm_name]
            color = get_arm_color(arm_name)

            ax.barh(
                y_positions + offset,
                values,
                height=bar_height,
                label=arm_name,
                color=color,
                edgecolor="white",
                linewidth=0.4,
                alpha=0.92,
            )

    with P("breakdown_styling"):
        # Build y-tick labels (questions only)
        y_labels = []
        for _, questions in struct_questions:
            for q_text, _ in questions:
                q_short = q_text[:45] + "..." if len(q_text) > 48 else q_text
                y_labels.append(q_short)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels, fontsize=9)

        # Add subtle background bands and structure labels
        band_colors = ["#f5f5f5", "#fafafa"]
        band_margin = 0.4  # Margin around each structure group

        for i, (start_pos, end_pos, label) in enumerate(structure_spans):
            # Subtle background band
            ax.axhspan(
                start_pos - band_margin, end_pos + band_margin,
                color=band_colors[i % 2],
                alpha=0.5, zorder=0
            )

            # Subtle left accent line
            ax.axhline(y=start_pos - band_margin, color="#e0e0e0",
                       linewidth=0.5, alpha=0.8, zorder=0)

            # Structure label - small, positioned at top-left
            ax.text(
                -0.015, start_pos - 0.25,
                label.lower(),
                ha="right", va="bottom",
                fontsize=7, fontweight="semibold",
                color="#888",
                transform=ax.get_yaxis_transform(),
            )

        # Set x-axis with more grid lines (minor ticks for grid only, not labels)
        ax.set_xlim(0, 105)
        ax.set_xlabel("Expected Structure Compliance (%)", fontsize=11)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticks([5, 10, 15, 20, 30, 35, 40, 45, 55, 60, 65, 70, 80, 85, 90, 95], minor=True)

        # No title - removed as requested

        # Grid and styling - clean, with minor gridlines
        ax.grid(True, axis="x", which="major", alpha=0.3, linestyle="-", linewidth=0.5, zorder=0)
        ax.grid(True, axis="x", which="minor", alpha=0.15, linestyle="-", linewidth=0.3, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Mark 50% threshold with a subtle vertical line
        ax.axvline(x=50, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6, zorder=1)

        # Invert y-axis (questions read top-to-bottom)
        ax.invert_yaxis()

    with P("breakdown_legend_meta"):
        # Add prompt as title at top center
        if metadata and metadata.get("prompt"):
            fig.suptitle(
                f'"{metadata["prompt"]}"',
                fontsize=12,
                style="italic",
                color="#333",
                y=0.98,
            )

        # Create hierarchical legend - vertically centered, smaller
        create_arm_legend(
            ax,
            arm_names,
            arm_descriptions,
            max_desc_length=35,
            fontsize=7,
            bbox_anchor=(1.04, 0.5),
            loc="center left",
        )

        # Model - bottom right: "Gen Model:" small monospace + model name bold
        if metadata and metadata.get("model"):
            ax.text(
                0.77, 0.012,
                "Gen Model:",
                transform=fig.transFigure,
                fontsize=6,
                fontfamily='monospace',
                verticalalignment="bottom",
                horizontalalignment="left",
                color="#999",
            )
            ax.text(
                0.84, 0.01,
                metadata['model'],
                transform=fig.transFigure,
                fontsize=9,
                fontweight='bold',
                verticalalignment="bottom",
                horizontalalignment="left",
                color="#444",
            )

        # Judge - above model
        if metadata and metadata.get("judge"):
            ax.text(
                0.77, 0.042,
                "Judge LLM:",
                transform=fig.transFigure,
                fontsize=6,
                fontfamily='monospace',
                verticalalignment="bottom",
                horizontalalignment="left",
                color="#999",
            )
            ax.text(
                0.84, 0.04,
                metadata['judge'],
                transform=fig.transFigure,
                fontsize=9,
                fontweight='bold',
                verticalalignment="bottom",
                horizontalalignment="left",
                color="#444",
            )

    with P("breakdown_save"):
        # Save with proper margins - more space on right for legend, metadata, and prompt
        plt.tight_layout()
        plt.subplots_adjust(left=0.28, right=0.50)
        save_figure(plt.gcf(), output_path)

    return output_path
