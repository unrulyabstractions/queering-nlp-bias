"""Bar plot visualization for system cores.

Visualizes core values (expected structure compliance) with one subplot per arm,
stacked vertically. One figure per generation method X weighting method combination.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from src.common.default_config import DEFAULT_WEIGHTING_METHOD
from src.common.profiler import P
from src.estimation.arm_types import get_arm_color, get_ordered_arms_for_plotting

from .viz_plot_utils import (
    annotate_bar_values,
    create_arm_legend,
    get_structure_color,
    save_figure,
    style_axis_clean,
)

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


def plot_cores_barplot(
    result: EstimationResult,
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create stacked bar plots of cores, one row per arm.

    Args:
        result: EstimationResult with arm data
        weighting_method: Name of weighting method (e.g., "prob", "inv-ppl")
        structure_labels: Labels for each structure dimension
        output_path: Where to save the PNG

    Returns:
        Path to saved file, or None if no data
    """
    # Extract cores for each arm in proper order
    arm_names = [arm.name for arm in result.arms]
    ordered_names = get_ordered_arms_for_plotting(arm_names)
    arm_data = []
    for name in ordered_names:
        arm = next((a for a in result.arms if a.name == name), None)
        if arm:
            core = arm.get_core(weighting_method)
            if core:
                arm_data.append((arm.name, core))

    if not arm_data:
        return None

    n_arms = len(arm_data)
    n_structures = len(structure_labels)

    # Setup figure with subplots stacked vertically
    fig_width = max(10, n_structures * 0.9 + 2)
    fig_height = 1.8 * n_arms + 1.2  # Height per arm + title space

    fig, axes = plt.subplots(
        n_arms, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    # Main title
    fig.suptitle(
        f"System Cores — {result.method} [{weighting_method}]",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(n_structures)
    bar_width = 0.7

    for i, (arm_name, core) in enumerate(arm_data):
        ax = axes[i]

        # Create bars with structure-specific colors
        colors = [get_structure_color(j) for j in range(len(core))]
        bars = ax.bar(
            x,
            core,
            bar_width,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

        # Add value labels on top of bars
        annotate_bar_values(ax, bars, core, fontsize=8, offset_points=(0, 2))

        # Styling for each subplot
        ax.set_ylabel(arm_name, fontsize=10, fontweight="bold", rotation=0, labelpad=50)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_xlim(-0.5, n_structures - 0.5)
        style_axis_clean(ax)

    # X-axis labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(structure_labels, fontsize=10)
    axes[-1].set_xlabel("Structure", fontsize=10)

    # Save
    save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0, 1, 0.96])
    return output_path


def plot_cores_comparison(
    result: EstimationResult,
    weighting_methods: list[str],
    structure_labels: list[str],
    output_path: Path,
    default_method: str = DEFAULT_WEIGHTING_METHOD,
    arm_descriptions: dict[str, str] | None = None,
    metadata: dict[str, str] | None = None,
) -> Path | None:
    """Create grouped bar plot comparing arms side-by-side, one row per weighting method.

    Shows root, trunk, and branches.

    Args:
        result: EstimationResult with arm data
        weighting_methods: List of weighting methods to plot (one row each)
        structure_labels: Labels for each structure dimension
        output_path: Where to save the PNG
        default_method: Which weighting method to show first (from default_config)
        arm_descriptions: Optional dict mapping arm names to conditioning text
        metadata: Optional dict with 'prompt', 'model', 'judge' keys for display

    Returns:
        Path to saved file, or None if no data
    """
    with P("comparison_data_prep"):
        # Show root, trunk, and branches (use arm_types for proper ordering)
        arm_names = [arm.name for arm in result.arms if arm.estimates]
        ordered_names = get_ordered_arms_for_plotting(arm_names)
        arms_to_plot = [
            next(a for a in result.arms if a.name == name)
            for name in ordered_names
        ]
        if not arms_to_plot or not weighting_methods:
            return None

        # Reorder weighting methods to put default first
        ordered_methods = [default_method] if default_method in weighting_methods else []
        ordered_methods.extend(m for m in weighting_methods if m != default_method)

        n_structures = len(structure_labels)
        n_arms = len(arms_to_plot)
        n_methods = len(ordered_methods)

    with P("comparison_figure_create"):
        # Setup figure with subplots stacked vertically
        # Width needs to accommodate many arms per structure group + large legend
        # Each structure group needs space proportional to n_arms
        width_per_structure = max(1.5, 0.25 * n_arms)
        fig_width = max(16, n_structures * width_per_structure + 10)  # Extra space for legend
        fig_height = 4.5 * n_methods + 1.5

        fig, axes = plt.subplots(
            n_methods, 1,
            figsize=(fig_width, fig_height),
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()

        # Main title
        fig.suptitle(
            f"System Cores by Arm — {result.method}",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

    with P("comparison_draw_bars"):
        x = np.arange(n_structures)
        # Bar width proportional to number of arms
        # With many arms, make bars thinner but ensure minimum visibility
        total_group_width = 0.85
        bar_width = max(0.05, total_group_width / n_arms)
        offsets = np.linspace(
            -total_group_width / 2 + bar_width / 2,
            total_group_width / 2 - bar_width / 2,
            n_arms
        )

        for row_idx, method in enumerate(ordered_methods):
            ax = axes[row_idx]

            # Draw bars for each arm
            for i, arm in enumerate(arms_to_plot):
                core = arm.get_core(method)
                if not core:
                    continue

                color = get_arm_color(arm.name)
                bars = ax.bar(
                    x + offsets[i],
                    core,
                    bar_width,
                    label=arm.name if row_idx == 0 else None,  # Legend only on first row
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.85,
                )

                # Add value labels on top of bars (scale font for many bars)
                annotate_bar_values(ax, bars, core, fontsize=7, n_bars_per_group=n_arms)

            # Styling for each subplot - method name as subtitle
            ax.set_title(f"[{method}]", fontsize=10, fontweight="bold", loc="left")
            ax.set_ylabel("Core", fontsize=9)
            ax.set_ylim(0, 1.15)
            ax.set_xlim(-0.6, n_structures - 0.4)  # Proper x-axis limits
            ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
            style_axis_clean(ax)

    with P("comparison_legend_meta"):
        # Legend outside plot with arm descriptions - vertically centered, smaller
        create_arm_legend(
            axes[0],
            ordered_names,
            arm_descriptions,
            max_desc_length=40,
            fontsize=10,
            bbox_anchor=(1.04, 0.5),  # Vertically centered
            loc="center left",
        )

        # X-axis labels only on bottom subplot
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(structure_labels, fontsize=9)
        axes[-1].set_xlabel("Structure", fontsize=10)

        # Add metadata (model, judge) - no prompt
        if metadata:
            # Model and Judge: "Label:" small monospace + value bold
            if metadata.get("model"):
                fig.text(
                    0.77, 0.012,
                    "Gen Model:",
                    fontsize=6,
                    fontfamily='monospace',
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    color="#999",
                )
                fig.text(
                    0.84, 0.01,
                    metadata['model'],
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    color="#444",
                )
            if metadata.get("judge"):
                fig.text(
                    0.77, 0.042,
                    "Judge LLM:",
                    fontsize=6,
                    fontfamily='monospace',
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    color="#999",
                )
                fig.text(
                    0.84, 0.04,
                    metadata['judge'],
                    fontsize=9,
                    fontweight='bold',
                    verticalalignment="bottom",
                    horizontalalignment="left",
                    color="#444",
                )

    with P("comparison_save"):
        # Save - leave room on right for legend
        save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0, 0.72, 0.96])

    return output_path


def plot_generation_comparison(
    results: list[EstimationResult],
    weighting_methods: list[str],
    structure_labels: list[str],
    output_path: Path,
    default_method: str = DEFAULT_WEIGHTING_METHOD,
) -> Path | None:
    """Create grouped bar plot comparing cores across generation methods.

    Shows cores for each arm (trunk, branch_1, etc.) as columns,
    with generation methods as grouped bars within each structure.
    One row per weighting method, with default method first.

    Args:
        results: EstimationResults from different generation methods
        weighting_methods: List of weighting methods to plot
        structure_labels: Labels for each structure dimension
        output_path: Where to save the PNG
        default_method: Which weighting method to show first (from default_config)

    Returns:
        Path to saved file, or None if no data
    """
    if len(results) < 2 or not weighting_methods:
        return None

    # Reorder weighting methods to put default first
    ordered_methods = [default_method] if default_method in weighting_methods else []
    ordered_methods.extend(m for m in weighting_methods if m != default_method)

    # Get all unique arm names across all results
    all_arm_names: set[str] = set()
    for result in results:
        for arm in result.arms:
            all_arm_names.add(arm.name)

    # Sort: root first, trunk second, then branches in order
    arm_names = get_ordered_arms_for_plotting(list(all_arm_names))
    n_arms = len(arm_names)
    n_structures = len(structure_labels)
    n_methods = len(ordered_methods)
    n_gen_methods = len(results)

    # Color palette for generation methods
    gen_method_colors = [
        "#4A90D9",  # blue
        "#E67E22",  # orange
        "#2ECC71",  # green
        "#9B59B6",  # purple
        "#E74C3C",  # red
    ]

    fig_width = max(12, n_arms * n_structures * 0.6 + 3)
    fig_height = 4.0 * n_methods + 1.5

    fig, axes = plt.subplots(
        n_methods, n_arms,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    fig.suptitle(
        "Generation Method Comparison",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(n_structures)
    bar_width = 0.8 / n_gen_methods
    offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_gen_methods)

    for row_idx, method in enumerate(ordered_methods):
        for col_idx, arm_name in enumerate(arm_names):
            ax = axes[row_idx, col_idx]

            for i, result in enumerate(results):
                # Get this arm from this result
                arm = next((a for a in result.arms if a.name == arm_name), None)
                if not arm:
                    continue

                core = arm.get_core(method)
                if not core:
                    continue

                color = gen_method_colors[i % len(gen_method_colors)]
                ax.bar(
                    x + offsets[i],
                    core,
                    bar_width,
                    label=result.method if row_idx == 0 and col_idx == 0 else None,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.85,
                )

            # Column titles (arm names) on top row
            if row_idx == 0:
                ax.set_title(arm_name.upper(), fontsize=11, fontweight="bold")

            # Row labels (weighting method) on left column
            if col_idx == 0:
                ax.set_ylabel(f"[{method}]", fontsize=10, fontweight="bold")

            ax.set_ylim(0, 1.15)
            ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
            style_axis_clean(ax)

    # Legend outside plot (upper right)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=9)

    # X-axis labels on bottom row
    for col_idx in range(n_arms):
        axes[-1, col_idx].set_xticks(x)
        axes[-1, col_idx].set_xticklabels(structure_labels, fontsize=9)

    save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0, 0.85, 0.96])
    return output_path
