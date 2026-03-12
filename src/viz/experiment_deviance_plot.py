"""Deviance and orientation visualization.

Shows deviance changes across arms and orientation vectors per branch.
Includes excess, deficit, mutual deviance and core diversity plots.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

BRANCH_COLORS = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C", "#1ABC9C", "#F39C12", "#8E44AD"]


def _format_deviance_value(val: float, precision: int = 4) -> str:
    """Format a deviance value, handling inf gracefully."""
    if math.isinf(val):
        return "∞" if val > 0 else "-∞"
    if math.isnan(val):
        return "NaN"
    return f"{val:.{precision}f}"


def _compute_label_positions(
    values: list[float],
    y_range: tuple[float, float],
    label_height_fraction: float = 0.15,
) -> list[float]:
    """Compute stacked y-positions for labels to prevent overlap.

    Each label takes about 2 lines of text, so we need generous spacing.

    Args:
        values: List of data y-values where labels should be placed
        y_range: (y_min, y_max) of the plot axis
        label_height_fraction: Fraction of y_range that each label occupies (15% default)

    Returns:
        List of y-positions (in data coords) where labels should be placed.
        These may differ from input values to avoid overlap.
    """
    if not values:
        return []

    n = len(values)
    if n == 1:
        return list(values)

    # Filter to finite values and track original indices
    indexed_vals = [(i, v) for i, v in enumerate(values) if math.isfinite(v)]
    if not indexed_vals:
        return list(values)

    # Compute minimum gap in data coordinates
    y_span = y_range[1] - y_range[0]
    min_gap = y_span * label_height_fraction

    # Sort by value (lowest first)
    indexed_vals.sort(key=lambda x: x[1])

    # Assign positions with stacking - always push up when close
    positions = list(values)  # Start with original values
    prev_label_bottom = float("-inf")

    for orig_idx, base_y in indexed_vals:
        # Check if this label would overlap with previous
        if base_y < prev_label_bottom + min_gap:
            # Stack above previous label
            new_y = prev_label_bottom + min_gap
            positions[orig_idx] = new_y
            prev_label_bottom = new_y
        else:
            positions[orig_idx] = base_y
            prev_label_bottom = base_y

    return positions


def plot_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plot showing deviance: trunk → branch for each branch."""
    trunk_arm = next((a for a in result.arms if a.name == "trunk"), None)
    branch_arms = [a for a in result.arms if a.name.startswith("branch_")]

    if not trunk_arm or not branch_arms:
        return None

    trunk_dev = trunk_arm.get_deviance_avg(weighting_method)

    # Handle inf trunk deviance
    if not math.isfinite(trunk_dev):
        return None

    # Adaptive figure width for many branches
    n_branches = len(branch_arms)
    fig_width = max(8, 6 + n_branches * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.array([0, 1])

    # Collect branch deviances for label positioning
    branch_devs = [arm.get_deviance_avg(weighting_method) for arm in branch_arms]

    # Filter out inf values for plotting
    plottable = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                 if math.isfinite(dev)]
    inf_branches = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                    if not math.isfinite(dev)]

    # Plot lines for finite branches first to establish y-axis range
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot(x, [trunk_dev, branch_dev], color=color, marker="o",
                markersize=10, linewidth=2.5, label=arm.name, alpha=0.9)

    # Get axis range for label positioning
    y_min, y_max = ax.get_ylim()
    y_range = (y_min, y_max)

    # Compute stacked label positions
    plottable_devs = [dev for _, _, dev in plottable]
    label_positions = _compute_label_positions(plottable_devs, y_range)

    # Add labels at computed positions
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        delta = branch_dev - trunk_dev
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        sign = "+" if delta >= 0 else ""
        label_y = label_positions[plot_idx]
        ax.annotate(
            f"E[∂|B] = {_format_deviance_value(branch_dev)}\nE[Δ∂] = {sign}{delta:.4f}",
            (1, label_y),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, fontweight="medium", color=color,
        )

    # Add legend entries for inf branches (not plotted)
    for orig_idx, arm, dev in inf_branches:
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot([], [], color=color, marker="o", markersize=10, linewidth=2.5,
                label=f"{arm.name} (E[∂]=∞)", alpha=0.9)

    # Trunk label - positioned above the line, left-aligned with plot area
    ax.annotate(
        f"E[∂|T] = {_format_deviance_value(trunk_dev)}",
        (0.02, trunk_dev), xytext=(0, 12), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, fontweight="medium", color="#555",
    )

    ax.axhline(y=trunk_dev, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_ylabel("E[∂] (Expected Deviance)", fontsize=11)
    ax.set_title(f"Deviance — {result.method} [{weighting_method}]", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["trunk", "branch"], fontsize=11, fontweight="medium")
    ax.set_xlim(-0.3, 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - (y_max - y_min) * 0.15, y_max + (y_max - y_min) * 0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_orientation_by_branch(
    result: EstimationResult,
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create bar plot showing E[θ|T] (orientation vs trunk) for branches."""
    from matplotlib.patches import Patch

    branch_arms = [a for a in result.arms if a.name.startswith("branch_")]
    if not branch_arms or not structure_labels:
        return None

    n_branches = len(branch_arms)
    n_structures = len(structure_labels)

    # Better proportioned figure
    fig_width = max(8, n_structures * 1.5 + 2)
    fig_height = 3.0 * n_branches + 1.0

    fig, axes = plt.subplots(
        n_branches, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Orientation E[θ|T] — {result.method} [{weighting_method}]",
        fontsize=13, fontweight="bold", y=0.98,
    )

    x = np.arange(n_structures)
    bar_width = 0.6

    for i, arm in enumerate(branch_arms):
        ax = axes[i]
        orientation = arm.get_orientation_avg(weighting_method) or [0.0] * n_structures
        orient_norm = arm.get_orientation_norm(weighting_method)

        colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in orientation]
        bars = ax.bar(x, orientation, bar_width, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

        # Value labels on bars
        for bar, val in zip(bars, orientation):
            height = bar.get_height()
            ax.annotate(
                f"{val:+.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=8, fontweight="medium",
            )

        # Title with branch name and norm
        ax.set_title(f"{arm.name.upper()}  (||θ|| = {orient_norm:.3f})", fontsize=10, fontweight="bold", loc="left")
        ax.axhline(y=0, color="#333", linewidth=1)
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_xlim(-0.5, n_structures - 0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add padding to y-axis for labels (after sharey takes effect)
    y_min, y_max = axes[0].get_ylim()
    padding = (y_max - y_min) * 0.12
    axes[0].set_ylim(y_min - padding, y_max + padding)

    # X-axis labels on bottom subplot
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(structure_labels, fontsize=10)
    axes[-1].set_xlabel("Structure", fontsize=10)

    # Proper legend
    legend_elements = [
        Patch(facecolor="#2ECC71", edgecolor="black", label="Over-compliance (+)"),
        Patch(facecolor="#E74C3C", edgecolor="black", label="Under-compliance (-)"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_excess_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plot showing excess deviance: trunk → branch for each branch.

    Excess deviance E[∂⁺] measures how much samples over-comply with structures.
    Uses Rényi divergence D_α(compliance || core).
    """
    trunk_arm = next((a for a in result.arms if a.name == "trunk"), None)
    branch_arms = [a for a in result.arms if a.name.startswith("branch_")]

    if not trunk_arm or not branch_arms:
        return None

    trunk_dev = trunk_arm.get_excess_deviance_avg(weighting_method)

    # Handle inf trunk deviance
    if not math.isfinite(trunk_dev):
        return None

    n_branches = len(branch_arms)
    fig_width = max(8, 6 + n_branches * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.array([0, 1])

    # Collect branch deviances for label positioning
    branch_devs = [arm.get_excess_deviance_avg(weighting_method) for arm in branch_arms]

    # Filter out inf values for plotting
    plottable = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                 if math.isfinite(dev)]
    inf_branches = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                    if not math.isfinite(dev)]

    # Plot lines first to establish y-axis range
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot(x, [trunk_dev, branch_dev], color=color, marker="o",
                markersize=10, linewidth=2.5, label=arm.name, alpha=0.9)

    # Compute stacked label positions
    plottable_devs = [dev for _, _, dev in plottable]
    y_min, y_max = ax.get_ylim()
    label_positions = _compute_label_positions(plottable_devs, (y_min, y_max))

    # Add labels at computed positions
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        delta = branch_dev - trunk_dev
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        sign = "+" if delta >= 0 else ""
        label_y = label_positions[plot_idx]
        ax.annotate(
            f"E[∂⁺|B] = {_format_deviance_value(branch_dev)}\nΔ = {sign}{delta:.4f}",
            (1, label_y),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, fontweight="medium", color=color,
        )

    # Add legend entries for inf branches
    for orig_idx, arm, dev in inf_branches:
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot([], [], color=color, marker="o", markersize=10, linewidth=2.5,
                label=f"{arm.name} (∞)", alpha=0.9)

    # Trunk label
    ax.annotate(
        f"E[∂⁺|T] = {_format_deviance_value(trunk_dev)}",
        (0.02, trunk_dev), xytext=(0, 12), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, fontweight="medium", color="#555",
    )

    ax.axhline(y=trunk_dev, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color="#2ECC71", linestyle=":", linewidth=1, alpha=0.5, label="neutral (1.0)")
    ax.set_ylabel("E[∂⁺] (Excess Deviance)", fontsize=11)
    ax.set_title(f"Excess Deviance — {result.method} [{weighting_method}]", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["trunk", "branch"], fontsize=11, fontweight="medium")
    ax.set_xlim(-0.3, 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(0.9, y_min - (y_max - y_min) * 0.15), y_max + (y_max - y_min) * 0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_deficit_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plot showing deficit deviance: trunk → branch for each branch.

    Deficit deviance E[∂⁻] measures how much samples under-comply with structures.
    Uses Rényi divergence D_α(core || compliance).
    """
    trunk_arm = next((a for a in result.arms if a.name == "trunk"), None)
    branch_arms = [a for a in result.arms if a.name.startswith("branch_")]

    if not trunk_arm or not branch_arms:
        return None

    trunk_dev = trunk_arm.get_deficit_deviance_avg(weighting_method)

    # Handle inf trunk deviance
    if not math.isfinite(trunk_dev):
        return None

    n_branches = len(branch_arms)
    fig_width = max(8, 6 + n_branches * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.array([0, 1])

    # Collect branch deviances for label positioning
    branch_devs = [arm.get_deficit_deviance_avg(weighting_method) for arm in branch_arms]

    # Filter out inf values for plotting
    plottable = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                 if math.isfinite(dev)]
    inf_branches = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                    if not math.isfinite(dev)]

    # Plot lines first to establish y-axis range
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot(x, [trunk_dev, branch_dev], color=color, marker="o",
                markersize=10, linewidth=2.5, label=arm.name, alpha=0.9)

    # Compute stacked label positions
    plottable_devs = [dev for _, _, dev in plottable]
    y_min, y_max = ax.get_ylim()
    label_positions = _compute_label_positions(plottable_devs, (y_min, y_max))

    # Add labels at computed positions
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        delta = branch_dev - trunk_dev
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        sign = "+" if delta >= 0 else ""
        label_y = label_positions[plot_idx]
        ax.annotate(
            f"E[∂⁻|B] = {_format_deviance_value(branch_dev)}\nΔ = {sign}{delta:.4f}",
            (1, label_y),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, fontweight="medium", color=color,
        )

    # Add legend entries for inf branches
    for orig_idx, arm, dev in inf_branches:
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot([], [], color=color, marker="o", markersize=10, linewidth=2.5,
                label=f"{arm.name} (∞)", alpha=0.9)

    # Trunk label
    ax.annotate(
        f"E[∂⁻|T] = {_format_deviance_value(trunk_dev)}",
        (0.02, trunk_dev), xytext=(0, 12), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, fontweight="medium", color="#555",
    )

    ax.axhline(y=trunk_dev, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color="#2ECC71", linestyle=":", linewidth=1, alpha=0.5, label="neutral (1.0)")
    ax.set_ylabel("E[∂⁻] (Deficit Deviance)", fontsize=11)
    ax.set_title(f"Deficit Deviance — {result.method} [{weighting_method}]", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["trunk", "branch"], fontsize=11, fontweight="medium")
    ax.set_xlim(-0.3, 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(max(0.9, y_min - (y_max - y_min) * 0.15), y_max + (y_max - y_min) * 0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_mutual_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plot showing mutual deviance: trunk → branch for each branch.

    Mutual deviance E[∂_M] uses Jensen-Shannon divergence (symmetric).
    Bounded between [1, 2] - always finite unlike KL-based measures.
    """
    trunk_arm = next((a for a in result.arms if a.name == "trunk"), None)
    branch_arms = [a for a in result.arms if a.name.startswith("branch_")]

    if not trunk_arm or not branch_arms:
        return None

    trunk_dev = trunk_arm.get_mutual_deviance_avg(weighting_method)

    # Handle inf trunk deviance (shouldn't happen for JS but just in case)
    if not math.isfinite(trunk_dev):
        return None

    n_branches = len(branch_arms)
    fig_width = max(8, 6 + n_branches * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, 6))
    x = np.array([0, 1])

    # Collect branch deviances for label positioning
    branch_devs = [arm.get_mutual_deviance_avg(weighting_method) for arm in branch_arms]

    # All should be finite for mutual deviance, but handle edge cases
    plottable = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                 if math.isfinite(dev)]
    inf_branches = [(i, arm, dev) for i, (arm, dev) in enumerate(zip(branch_arms, branch_devs))
                    if not math.isfinite(dev)]

    # Plot lines first
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot(x, [trunk_dev, branch_dev], color=color, marker="o",
                markersize=10, linewidth=2.5, label=arm.name, alpha=0.9)

    # Compute stacked label positions (mutual deviance has fixed y-range [0.95, 2.05])
    plottable_devs = [dev for _, _, dev in plottable]
    label_positions = _compute_label_positions(plottable_devs, (0.95, 2.05))

    # Add labels at computed positions
    for plot_idx, (orig_idx, arm, branch_dev) in enumerate(plottable):
        delta = branch_dev - trunk_dev
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        sign = "+" if delta >= 0 else ""
        label_y = label_positions[plot_idx]
        ax.annotate(
            f"E[∂_M|B] = {_format_deviance_value(branch_dev)}\nΔ = {sign}{delta:.4f}",
            (1, label_y),
            xytext=(10, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9, fontweight="medium", color=color,
        )

    # Add legend entries for inf branches
    for orig_idx, arm, dev in inf_branches:
        color = BRANCH_COLORS[orig_idx % len(BRANCH_COLORS)]
        ax.plot([], [], color=color, marker="o", markersize=10, linewidth=2.5,
                label=f"{arm.name} (∞)", alpha=0.9)

    # Trunk label
    ax.annotate(
        f"E[∂_M|T] = {_format_deviance_value(trunk_dev)}",
        (0.02, trunk_dev), xytext=(0, 12), textcoords="offset points",
        ha="left", va="bottom", fontsize=9, fontweight="medium", color="#555",
    )

    ax.axhline(y=trunk_dev, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=1.0, color="#2ECC71", linestyle=":", linewidth=1, alpha=0.5, label="neutral (1.0)")
    ax.axhline(y=2.0, color="#E74C3C", linestyle=":", linewidth=1, alpha=0.5, label="max (2.0)")
    ax.set_ylabel("E[∂_M] (Mutual Deviance)", fontsize=11)
    ax.set_title(f"Mutual Deviance — {result.method} [{weighting_method}]", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["trunk", "branch"], fontsize=11, fontweight="medium")
    ax.set_xlim(-0.3, 1.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Fixed y-axis for mutual deviance (bounded [1, 2])
    ax.set_ylim(0.95, 2.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def plot_core_diversity_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create bar plot showing core diversity (Hill D_1) for each arm.

    Core diversity measures the effective number of structures.
    D_1 = exp(H(core_normalized)) ∈ [1, n] where n = number of structures.
    """
    arms_to_plot = [a for a in result.arms if a.name in ("trunk", "all_arms") or a.name.startswith("branch_")]

    if not arms_to_plot:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(arms_to_plot))
    bar_width = 0.6

    diversities = [arm.get_core_diversity(weighting_method) for arm in arms_to_plot]
    names = [arm.name for arm in arms_to_plot]

    # Color by arm type
    colors = []
    for arm in arms_to_plot:
        if arm.name == "trunk":
            colors.append("#4A90D9")
        elif arm.name == "all_arms":
            colors.append("#888888")
        else:
            idx = int(arm.name.split("_")[1]) - 1 if "_" in arm.name else 0
            colors.append(BRANCH_COLORS[idx % len(BRANCH_COLORS)])

    bars = ax.bar(x, diversities, bar_width, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)

    # Value labels on bars
    for bar, val in zip(bars, diversities):
        height = bar.get_height()
        ax.annotate(
            f"{val:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    # Reference line at 1 (minimum diversity)
    ax.axhline(y=1.0, color="#E74C3C", linestyle=":", linewidth=1.5, alpha=0.7, label="min diversity (1)")

    # Get max possible diversity (number of structures)
    first_arm = arms_to_plot[0]
    n_structures = len(first_arm.get_core(weighting_method))
    if n_structures > 0:
        ax.axhline(y=n_structures, color="#2ECC71", linestyle=":", linewidth=1.5, alpha=0.7,
                   label=f"max diversity ({n_structures})")

    ax.set_ylabel("Core Diversity (D₁)", fontsize=11)
    ax.set_title(f"Core Diversity — {result.method} [{weighting_method}]", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11, fontweight="medium")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-axis to start at 0
    y_max = max(diversities) if diversities else 1
    ax.set_ylim(0, max(y_max * 1.2, n_structures * 1.1 if n_structures > 0 else 2))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path
