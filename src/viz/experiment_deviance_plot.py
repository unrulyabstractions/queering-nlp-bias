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

from src.estimation.arm_types import (
    ArmKind,
    classify_arm,
    get_arm_ancestry,
    get_arm_color,
    get_downstream_arms,
    get_ordered_arms_for_plotting,
    has_downstream_arms,
)

from .viz_plot_utils import add_dense_grid, annotate_bar_values, is_camera_ready, save_figure, style_axis_clean

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


def _compute_stagger_offsets(arm_names: list[str]) -> dict[str, float]:
    """Compute x-axis offsets for staggering branches and twigs.

    Root and trunk stay at their base positions (offset=0).
    Branches are spread around their base position.
    Twigs are spread around their base position.

    Returns:
        Dict mapping arm name to x offset (e.g., -0.15, 0, +0.15)
    """
    offsets: dict[str, float] = {}

    # Group arms by kind
    branches = [n for n in arm_names if n.startswith("branch_")]
    twigs = [n for n in arm_names if n.startswith("twig_")]

    # Root and trunk: no offset
    offsets["root"] = 0.0
    offsets["trunk"] = 0.0

    # Stagger branches around x=2
    n_branches = len(branches)
    if n_branches > 1:
        spread = 0.25  # Total spread width
        for i, name in enumerate(sorted(branches)):
            # Center the offsets: [-spread/2, ..., +spread/2]
            offset = -spread / 2 + (i / (n_branches - 1)) * spread if n_branches > 1 else 0
            offsets[name] = offset
    else:
        for name in branches:
            offsets[name] = 0.0

    # Stagger twigs around x=3
    n_twigs = len(twigs)
    if n_twigs > 1:
        spread = 0.35  # Wider spread for twigs (usually more of them)
        for i, name in enumerate(sorted(twigs)):
            offset = -spread / 2 + (i / (n_twigs - 1)) * spread if n_twigs > 1 else 0
            offsets[name] = offset
    else:
        for name in twigs:
            offsets[name] = 0.0

    return offsets


def _plot_deviance_trajectories(
    result: "EstimationResult",
    weighting_method: str,
    metric_getter: str,
    metric_name: str,
    title: str,
    output_path: Path,
    reference_line: float | None = None,
    reference_label: str | None = None,
) -> Path | None:
    """Generic trajectory plot for deviance metrics.

    Each arm draws a line through its ancestry showing how the metric
    evolves as conditioning text is added.

    Args:
        result: Estimation result
        weighting_method: Weighting method name
        metric_getter: Method name to call on arm (e.g., "get_deviance_avg")
        metric_name: Display name for metric (e.g., "E[∂]")
        title: Plot title
        output_path: Where to save
        reference_line: Optional y-value for reference line
        reference_label: Label for reference line
    """
    # Build metric lookup
    metric_by_name: dict[str, float] = {}
    variance_by_name: dict[str, float] = {}
    for arm in result.arms:
        getter = getattr(arm, metric_getter)
        val = getter(weighting_method)
        if math.isfinite(val):
            metric_by_name[arm.name] = val
            # Get variance if available (for deviance metrics)
            var_val = arm.get_deviance_var(weighting_method)
            if math.isfinite(var_val):
                variance_by_name[arm.name] = var_val

    if not metric_by_name:
        return None

    # Get arms ordered
    arm_names = [a.name for a in result.arms]
    ordered_names = get_ordered_arms_for_plotting(arm_names)

    # Find max ancestry depth
    max_depth = 0
    for name in ordered_names:
        ancestry = get_arm_ancestry(name)
        max_depth = max(max_depth, len(ancestry))

    fig, ax = plt.subplots(figsize=(12, 7))
    stage_labels = ["root", "trunk", "branch", "twig"][:max_depth]

    # Compute x-offsets for staggering branches and twigs
    x_offsets = _compute_stagger_offsets(ordered_names)

    # Plot each arm's trajectory
    for arm_name in ordered_names:
        if arm_name not in metric_by_name:
            continue

        ancestry = get_arm_ancestry(arm_name)
        color = get_arm_color(arm_name)
        kind = classify_arm(arm_name)

        # Get metric values and variances along ancestry
        # Apply x offset for staggering at the final point
        x_positions = []
        y_values = []
        y_stds = []
        arm_offset = x_offsets.get(arm_name, 0.0)

        for i, anc_name in enumerate(ancestry):
            if anc_name in metric_by_name:
                # Apply stagger offset only to this arm's own point (final in ancestry)
                is_own_point = (anc_name == arm_name)
                x_pos = i + (arm_offset if is_own_point else 0.0)
                x_positions.append(x_pos)
                y_values.append(metric_by_name[anc_name])
                # Get std dev (sqrt of variance) for error bars
                var = variance_by_name.get(anc_name, 0.0)
                y_stds.append(math.sqrt(var) if var > 0 else 0.0)

        if not x_positions:
            continue

        # Plot line (connects through staggered final point)
        if len(x_positions) > 1:
            ax.plot(x_positions, y_values, '-', color=color, linewidth=2.5, alpha=0.7)

        # Plot variance bars (background, semi-transparent)
        for x, y, std in zip(x_positions, y_values, y_stds):
            if std > 0:
                ax.errorbar(
                    x, y, yerr=std,
                    fmt='none', ecolor=color, alpha=0.3,
                    capsize=4, capthick=1.5, elinewidth=2,
                    zorder=4,
                )

        # Plot points
        for i, (x, y) in enumerate(zip(x_positions, y_values)):
            is_final = (i == len(x_positions) - 1)
            if is_final:
                marker_size = 140 if kind == ArmKind.ROOT else 100
                ax.scatter([x], [y], s=marker_size, c=[color], edgecolors='white',
                          linewidths=1.5, zorder=10, label=arm_name)
                # Smaller, cleaner value label
                ax.annotate(
                    f"{y:.2f}",
                    xy=(x, y), xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7, fontweight="medium", color=color,
                    alpha=0.9,
                )
            else:
                ax.scatter([x], [y], s=35, c=[color], edgecolors='white',
                          linewidths=0.8, alpha=0.5, zorder=5)

    # Reference line
    if reference_line is not None:
        ax.axhline(y=reference_line, color="#888", linestyle=":", linewidth=1.5,
                   alpha=0.7, label=reference_label or f"{reference_line}")

    # Styling
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel(metric_name, fontsize=10)
    ax.set_title(f"{title} — {result.method} [{weighting_method}]",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(max_depth))
    ax.set_xticklabels(stage_labels, fontsize=10)

    # Legend: compact, outside plot area
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1),
        fontsize=8, framealpha=0.95, edgecolor="#ddd",
        handlelength=1.2, handletextpad=0.4,
    )

    # Dense grid for better readability
    add_dense_grid(ax, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Y-axis padding
    all_vals = list(metric_by_name.values())
    if all_vals:
        y_min, y_max = min(all_vals), max(all_vals)
        padding = (y_max - y_min) * 0.18 if y_max > y_min else 0.1
        ax.set_ylim(y_min - padding, y_max + padding * 2.5)

    save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0, 0.82, 1])
    return output_path


def plot_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create trajectory plot showing E[∂|self] evolution through conditioning stages.

    Shows all arms (root, trunk, branches, twigs) as trajectories through
    their conditioning ancestry.
    """
    return _plot_deviance_trajectories(
        result=result,
        weighting_method=weighting_method,
        metric_getter="get_deviance_avg",
        metric_name="E[∂|self]",
        title="Expected Deviance",
        output_path=output_path,
    )


def plot_orientation_for_reference(
    result: EstimationResult,
    reference_arm_name: str,
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create bar plot showing orientation of downstream arms relative to a reference.

    Only shows arms that are downstream in the conditioning hierarchy:
    - root: shows trunk, branches, twigs
    - trunk: shows branches, twigs
    - branch_N: shows only twigs of that branch
    - twig: nothing downstream (no plot created)

    Args:
        result: Estimation result
        reference_arm_name: Name of reference arm (root, trunk, branch_1, etc.)
        weighting_method: Weighting method to use
        structure_labels: Labels for structures
        output_path: Where to save the plot
    """
    # Get reference arm
    ref_arm = next((a for a in result.arms if a.name == reference_arm_name), None)
    if not ref_arm:
        return None

    ref_core = ref_arm.get_core(weighting_method)
    if not ref_core:
        return None

    # Get only downstream arms (not all other arms!)
    all_arm_names = [a.name for a in result.arms]
    downstream_names = get_downstream_arms(reference_arm_name, all_arm_names)

    if not downstream_names:
        # No downstream arms (e.g., reference is a twig)
        return None

    # Get arm objects for downstream arms
    other_arms = [a for a in result.arms if a.name in downstream_names]
    if not other_arms or not structure_labels:
        return None

    n_arms = len(other_arms)
    n_structures = len(structure_labels)

    # Figure size - wider for better bar visibility
    fig_width = max(12, n_structures * 2.0 + 3)
    fig_height = 3.5 * n_arms + 1.5

    fig, axes = plt.subplots(
        n_arms, 1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.flatten()

    fig.suptitle(
        f"Orientation E[θ|{reference_arm_name}] — {result.method} [{weighting_method}]",
        fontsize=14, fontweight="bold", y=0.98,
    )

    x = np.arange(n_structures)
    bar_width = 0.75  # Wider bars for better visibility

    for i, arm in enumerate(other_arms):
        ax = axes[i]

        # Use pre-computed orientation based on reference type
        arm_kind = classify_arm(arm.name)
        if reference_arm_name == "root":
            orientation = arm.get_orientation_from_root(weighting_method)
            orient_norm = arm.get_orientation_norm_from_root(weighting_method)
        elif reference_arm_name == "trunk":
            orientation = arm.get_orientation_from_trunk(weighting_method)
            orient_norm = arm.get_orientation_norm_from_trunk(weighting_method)
        elif arm_kind == ArmKind.TWIG:
            # For twigs relative to parent branch, use parent orientation
            orientation = arm.get_orientation_from_parent(weighting_method)
            orient_norm = arm.get_orientation_norm_from_parent(weighting_method)
        else:
            # Non-twig arm relative to a branch - this shouldn't happen
            # (branches can only have twigs as downstream, not other branches)
            raise ValueError(
                f"Cannot compute orientation for arm '{arm.name}' (kind={arm_kind}) "
                f"relative to '{reference_arm_name}'. Only twigs can be downstream of branches."
            )

        # Validate orientation data exists
        if not orientation:
            raise ValueError(
                f"Missing orientation data for arm '{arm.name}' relative to '{reference_arm_name}'. "
                f"Ensure estimation pipeline computed orientation fields."
            )

        # Use structure colors from palette (same as evolution tree)
        from .viz_plot_utils import get_structure_color
        colors = [get_structure_color(j) for j in range(len(orientation))]
        bars = ax.bar(x, orientation, bar_width, color=colors, edgecolor="white", linewidth=1.0, alpha=0.9)

        # Always show value labels on bars
        for bar, val in zip(bars, orientation):
            height = bar.get_height()
            val_str = "0" if abs(val) < 0.005 else f"{val:+.2f}"
            ax.annotate(
                val_str,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -4),
                textcoords="offset points",
                ha="center", va="bottom" if height >= 0 else "top",
                fontsize=9, fontweight="bold", color="#333",
            )

        # Title with arm name (colored) and norm
        arm_color = get_arm_color(arm.name)
        ax.set_title(
            f"{arm.name.upper()}  (||θ|| = {orient_norm:.3f})",
            fontsize=11, fontweight="bold", loc="left", color=arm_color
        )
        ax.axhline(y=0, color="#333", linewidth=1.2)
        ax.set_xlim(-0.5, n_structures - 0.5)
        # Use dense grid for better readability
        style_axis_clean(ax, dense_grid=True, grid_axis="y")

    # Add padding to y-axis
    y_min, y_max = axes[0].get_ylim()
    padding = (y_max - y_min) * 0.12
    axes[0].set_ylim(y_min - padding, y_max + padding)

    # X-axis labels on bottom subplot - wrap long labels
    axes[-1].set_xticks(x)
    # Wrap long structure labels
    wrapped_labels = [label if len(label) <= 25 else label[:22] + "..." for label in structure_labels]
    axes[-1].set_xticklabels(wrapped_labels, fontsize=10, rotation=15, ha="right")
    axes[-1].set_xlabel("Structure", fontsize=11, fontweight="medium")

    # Add more vertical padding
    plt.subplots_adjust(hspace=0.35)
    save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0.02, 1, 0.96])
    return output_path


def plot_orientation_by_branch(
    result: EstimationResult,
    weighting_method: str,
    structure_labels: list[str],
    output_dir: Path,
) -> list[Path]:
    """Create orientation plots for arms that have downstream children.

    Only creates plots for arms with downstream children in the hierarchy:
    - orientation_root.png - shows trunk, branches, twigs relative to root
    - orientation_trunk.png - shows branches, twigs relative to trunk
    - orientation_branch_1.png - shows twigs of branch_1 (if any)
    - NO plots for twigs (nothing downstream)

    Args:
        result: Estimation result
        weighting_method: Weighting method to use
        structure_labels: Labels for structures
        output_dir: Directory to save plots

    Returns:
        List of created file paths
    """
    created_files: list[Path] = []

    # Get all arm names
    all_arm_names = [a.name for a in result.arms]

    # Only create orientation plots for arms that have downstream children
    reference_arms = [
        name for name in all_arm_names
        if has_downstream_arms(name, all_arm_names)
    ]

    # Create orientation subfolder
    orientation_dir = output_dir / "orientation"
    orientation_dir.mkdir(parents=True, exist_ok=True)

    for ref_name in reference_arms:
        # Save as orientation/{ref_name}.png (not orientation_{ref_name}.png)
        output_path = orientation_dir / f"{ref_name}.png"
        path = plot_orientation_for_reference(
            result, ref_name, weighting_method, structure_labels, output_path
        )
        if path:
            created_files.append(path)

    return created_files


def plot_excess_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create trajectory plot showing E[∂⁺] evolution through conditioning stages.

    Excess deviance measures over-compliance with structures (Rényi divergence).
    Shows all arms as trajectories through conditioning ancestry.
    """
    return _plot_deviance_trajectories(
        result=result,
        weighting_method=weighting_method,
        metric_getter="get_excess_deviance_avg",
        metric_name="E[∂⁺]",
        title="Excess Deviance",
        output_path=output_path,
    )


def plot_deficit_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create trajectory plot showing E[∂⁻] evolution through conditioning stages.

    Deficit deviance measures under-compliance with structures (Rényi divergence).
    Shows all arms as trajectories through conditioning ancestry.
    """
    return _plot_deviance_trajectories(
        result=result,
        weighting_method=weighting_method,
        metric_getter="get_deficit_deviance_avg",
        metric_name="E[∂⁻]",
        title="Deficit Deviance",
        output_path=output_path,
    )


def plot_mutual_deviance_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create trajectory plot showing E[∂_M] evolution through conditioning stages.

    Mutual deviance uses Jensen-Shannon divergence (symmetric, bounded [1, 2]).
    Shows all arms as trajectories through conditioning ancestry.
    """
    return _plot_deviance_trajectories(
        result=result,
        weighting_method=weighting_method,
        metric_getter="get_mutual_deviance_avg",
        metric_name="E[∂_M]",
        title="Mutual Deviance",
        output_path=output_path,
    )


def plot_core_diversity_by_arm(
    result: EstimationResult,
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create trajectory plot showing diversity evolution through conditioning stages.

    Each arm draws a line through its ancestry:
    - Root: single point at position 0
    - Trunk: line from root (pos 0) to trunk (pos 1)
    - Branch_1: line from root → trunk → branch_1 (pos 0 → 1 → 2)
    - Twig_1_b1: line root → trunk → branch_1 → twig (pos 0 → 1 → 2 → 3)

    Shows how diversity evolves as conditioning text is added.
    """
    # Build diversity lookup
    diversity_by_name: dict[str, float] = {}
    for arm in result.arms:
        diversity_by_name[arm.name] = arm.get_core_diversity(weighting_method)

    if not diversity_by_name:
        return None

    # Get arms ordered
    arm_names = [a.name for a in result.arms]
    ordered_names = get_ordered_arms_for_plotting(arm_names)

    # Find max ancestry depth for x-axis
    max_depth = 0
    for name in ordered_names:
        ancestry = get_arm_ancestry(name)
        max_depth = max(max_depth, len(ancestry))

    fig, ax = plt.subplots(figsize=(12, 7))

    # X-axis labels for stages
    stage_labels = ["root", "trunk", "branch", "twig"][:max_depth]

    # Plot each arm's trajectory through its ancestry
    for arm_name in ordered_names:
        ancestry = get_arm_ancestry(arm_name)
        color = get_arm_color(arm_name)
        kind = classify_arm(arm_name)

        # Get diversity values along ancestry
        x_positions = []
        y_values = []
        for i, anc_name in enumerate(ancestry):
            if anc_name in diversity_by_name:
                x_positions.append(i)
                y_values.append(diversity_by_name[anc_name])

        if not x_positions:
            continue

        # Plot line connecting ancestry points
        if len(x_positions) > 1:
            ax.plot(x_positions, y_values, '-', color=color, linewidth=2.5, alpha=0.7)

        # Plot points - bigger for final arm, smaller for ancestors
        for i, (x, y) in enumerate(zip(x_positions, y_values)):
            is_final = (i == len(x_positions) - 1)
            if is_final:
                # Final point is bigger with label
                marker_size = 180 if kind == ArmKind.ROOT else 120
                ax.scatter([x], [y], s=marker_size, c=[color], edgecolors='white',
                          linewidths=2, zorder=10)
                # Label with diversity value
                ax.annotate(
                    f"{y:.2f}",
                    xy=(x, y), xytext=(5, 8),
                    textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color,
                )
            else:
                # Ancestor points are smaller
                ax.scatter([x], [y], s=40, c=[color], edgecolors='white',
                          linewidths=1, alpha=0.6, zorder=5)

    # Get max possible diversity (n_structures)
    first_arm = next((a for a in result.arms), None)
    n_structures = 0
    if first_arm:
        core = first_arm.get_core(weighting_method)
        n_structures = len(core) if core else 0

    # Y-axis limits - include both data max AND reference line max
    all_divs = list(diversity_by_name.values())
    data_max = max(all_divs) if all_divs else 1
    y_max = max(data_max, n_structures) * 1.15  # Include max reference line
    ax.set_ylim(0, y_max)

    # Reference lines (now guaranteed to be within plot)
    ax.axhline(y=1.0, color="#E74C3C", linestyle=":", linewidth=1.5, alpha=0.7)
    if n_structures > 0:
        ax.axhline(y=n_structures, color="#2ECC71", linestyle=":", linewidth=1.5, alpha=0.7)

    # Styling
    ax.set_xlabel("Token Position", fontsize=10)
    ax.set_ylabel("Core Diversity (D₁)", fontsize=10)
    ax.set_title(f"Diversity Evolution — {result.method} [{weighting_method}]",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(max_depth))
    ax.set_xticklabels(stage_labels, fontsize=10)

    # Create arm legend (outside plot, upper)
    from matplotlib.lines import Line2D
    arm_legend_elements = []
    for arm_name in ordered_names:
        color = get_arm_color(arm_name)
        arm_legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                   markersize=8, label=arm_name)
        )

    arm_legend = ax.legend(
        handles=arm_legend_elements,
        loc="upper left", bbox_to_anchor=(1.01, 1),
        fontsize=8, framealpha=0.95, edgecolor="#ddd",
    )
    ax.add_artist(arm_legend)

    # Create separate reference line legend (below arm legend)
    ref_legend_elements = [
        Line2D([0], [0], color='#E74C3C', linestyle=':', linewidth=1.5, label='min (1)'),
    ]
    if n_structures > 0:
        ref_legend_elements.append(
            Line2D([0], [0], color='#2ECC71', linestyle=':', linewidth=1.5,
                   label=f'max ({n_structures})')
        )

    ax.legend(
        handles=ref_legend_elements,
        loc="lower left", bbox_to_anchor=(1.01, 0),
        fontsize=8, framealpha=0.95, edgecolor="#ddd",
    )

    # Dense grid for better readability
    add_dense_grid(ax, axis="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    save_figure(plt.gcf(), output_path, tight_layout_rect=[0, 0, 0.82, 1])
    return output_path
