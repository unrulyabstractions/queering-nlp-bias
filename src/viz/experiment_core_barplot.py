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

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

# Colors for structures (consistent across all arms)
STRUCTURE_COLORS = [
    "#5DA5DA",  # blue
    "#FAA43A",  # orange
    "#60BD68",  # green
    "#F17CB0",  # pink
    "#B2912F",  # brown
    "#B276B2",  # purple
    "#DECF3F",  # yellow
    "#F15854",  # red
    "#4D4D4D",  # gray
]


def get_structure_color(idx: int) -> str:
    """Get color for a structure by index."""
    return STRUCTURE_COLORS[idx % len(STRUCTURE_COLORS)]


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
    # Extract cores for each arm (excluding all_arms)
    arm_data = []
    for arm in result.arms:
        if arm.name == "all_arms":
            continue
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
        for bar, val in zip(bars, core):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#333333",
                fontweight="medium",
            )

        # Styling for each subplot
        ax.set_ylabel(arm_name, fontsize=10, fontweight="bold", rotation=0, labelpad=50)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_xlim(-0.5, n_structures - 0.5)

        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # X-axis labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(structure_labels, fontsize=10)
    axes[-1].set_xlabel("Structure", fontsize=10)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


# Colors for arms in comparison plot
ARM_COLORS = {
    "trunk": "#888888",
    "branch_1": "#4A90D9",
    "branch_2": "#E67E22",
    "branch_3": "#2ECC71",
    "branch_4": "#9B59B6",
    "branch_5": "#E74C3C",
}


def get_arm_color(arm_name: str) -> str:
    """Get color for an arm by name."""
    if arm_name in ARM_COLORS:
        return ARM_COLORS[arm_name]
    # Fallback for additional branches
    return "#999999"


def plot_cores_comparison(
    result: EstimationResult,
    weighting_methods: list[str],
    structure_labels: list[str],
    output_path: Path,
    default_method: str = DEFAULT_WEIGHTING_METHOD,
) -> Path | None:
    """Create grouped bar plot comparing arms side-by-side, one row per weighting method.

    Shows trunk and branches (excludes all_arms). For trunk vs all_arms,
    use plot_trunk_vs_all_arms().

    Args:
        result: EstimationResult with arm data
        weighting_methods: List of weighting methods to plot (one row each)
        structure_labels: Labels for each structure dimension
        output_path: Where to save the PNG
        default_method: Which weighting method to show first (default: "prob")

    Returns:
        Path to saved file, or None if no data
    """
    # Filter out all_arms - only show trunk and branches
    arms_to_plot = [arm for arm in result.arms if arm.name != "all_arms" and arm.estimates]
    if not arms_to_plot or not weighting_methods:
        return None

    # Reorder weighting methods to put default first
    ordered_methods = [default_method] if default_method in weighting_methods else []
    ordered_methods.extend(m for m in weighting_methods if m != default_method)

    n_structures = len(structure_labels)
    n_arms = len(arms_to_plot)
    n_methods = len(ordered_methods)

    # Setup figure with subplots stacked vertically
    fig_width = max(8, n_structures * 0.8 + 2)
    fig_height = 4.0 * n_methods + 0.8

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

    x = np.arange(n_structures)
    bar_width = 0.8 / n_arms
    offsets = np.linspace(-0.4 + bar_width / 2, 0.4 - bar_width / 2, n_arms)

    for row_idx, method in enumerate(ordered_methods):
        ax = axes[row_idx]

        # Draw bars for each arm (excluding all_arms)
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

            # Add value labels on top of bars
            for bar, val in zip(bars, core):
                height = bar.get_height()
                ax.annotate(
                    f"{val:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#333333",
                )

        # Styling for each subplot - method name as subtitle
        ax.set_title(f"[{method}]", fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Core", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend outside plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=9)

    # X-axis labels only on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(structure_labels, fontsize=9)
    axes[-1].set_xlabel("Structure", fontsize=10)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

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
        default_method: Which weighting method to show first (default: "prob")

    Returns:
        Path to saved file, or None if no data
    """
    if len(results) < 2 or not weighting_methods:
        return None

    # Reorder weighting methods to put default first
    ordered_methods = [default_method] if default_method in weighting_methods else []
    ordered_methods.extend(m for m in weighting_methods if m != default_method)

    # Get all unique arm names across all results (excluding all_arms)
    all_arm_names: set[str] = set()
    for result in results:
        for arm in result.arms:
            if arm.name != "all_arms":
                all_arm_names.add(arm.name)

    # Sort: trunk first, then branches in order
    arm_names = sorted(all_arm_names, key=lambda x: (0, "") if x == "trunk" else (1, x))
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
                bars = ax.bar(
                    x + offsets[i],
                    core,
                    bar_width,
                    label=result.method if row_idx == 0 and col_idx == 0 else None,
                    color=color,
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.85,
                )

                # Add value labels on bars
                for bar, val in zip(bars, core):
                    height = bar.get_height()
                    ax.annotate(
                        f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, color="#333333",
                    )

            # Column titles (arm names) on top row
            if row_idx == 0:
                ax.set_title(arm_name.upper(), fontsize=11, fontweight="bold")

            # Row labels (weighting method) on left column
            if col_idx == 0:
                ax.set_ylabel(f"[{method}]", fontsize=10, fontweight="bold")

            ax.set_ylim(0, 1.15)
            ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
            ax.grid(axis="y", alpha=0.3, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    # Legend outside plot (upper right)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=9)

    # X-axis labels on bottom row
    for col_idx in range(n_arms):
        axes[-1, col_idx].set_xticks(x)
        axes[-1, col_idx].set_xticklabels(structure_labels, fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_trunk_vs_all_arms(
    result: EstimationResult,
    weighting_methods: list[str],
    structure_labels: list[str],
    output_path: Path,
    default_method: str = DEFAULT_WEIGHTING_METHOD,
) -> Path | None:
    """Create comparison plot: trunk (arm-only) vs all_arms (pooled).

    Shows how the trunk-only core differs from the pooled core.

    Args:
        result: EstimationResult with arm data
        weighting_methods: List of weighting methods to plot
        structure_labels: Labels for each structure dimension
        output_path: Where to save the PNG
        default_method: Which weighting method to show first (default: "prob")

    Returns:
        Path to saved file, or None if no data
    """
    # Find trunk and all_arms
    trunk_arm = next((a for a in result.arms if a.name == "trunk"), None)
    all_arms = next((a for a in result.arms if a.name == "all_arms"), None)

    if not trunk_arm or not all_arms or not weighting_methods:
        return None

    # Reorder weighting methods to put default first
    ordered_methods = [default_method] if default_method in weighting_methods else []
    ordered_methods.extend(m for m in weighting_methods if m != default_method)

    n_structures = len(structure_labels)
    n_methods = len(ordered_methods)

    fig_width = max(8, n_structures * 0.8 + 2)
    fig_height = 4.0 * n_methods + 0.8

    fig, axes = plt.subplots(
        n_methods, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Trunk vs All Arms — {result.method}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    x = np.arange(n_structures)
    bar_width = 0.35
    colors = {"trunk": "#4A90D9", "all_arms": "#E67E22"}

    for row_idx, method in enumerate(ordered_methods):
        ax = axes[row_idx]

        trunk_core = trunk_arm.get_core(method)
        all_core = all_arms.get_core(method)

        if not trunk_core or not all_core:
            continue

        # Draw bars side by side
        bars1 = ax.bar(
            x - bar_width / 2, trunk_core, bar_width,
            label="trunk" if row_idx == 0 else None,
            color=colors["trunk"], edgecolor="black", linewidth=0.5, alpha=0.85,
        )
        bars2 = ax.bar(
            x + bar_width / 2, all_core, bar_width,
            label="all_arms" if row_idx == 0 else None,
            color=colors["all_arms"], edgecolor="black", linewidth=0.5, alpha=0.85,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color="#333333",
                )

        ax.set_title(f"[{method}]", fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("Core", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.axhline(y=0.5, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend outside plot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=9)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(structure_labels, fontsize=9)
    axes[-1].set_xlabel("Structure", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


