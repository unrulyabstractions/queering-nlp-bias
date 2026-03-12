"""Generalized cores and deviance visualizations.

Shows different statistical variants (standard, uniform, mode, etc.) with their
(q, r) parameters as heatmaps and line plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from src.estimation.arm_types import get_arm_color, get_ordered_arms_for_plotting

from .viz_plot_utils import save_figure, style_axis_clean

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult


def _group_arms_into_rows(arm_names: list[str]) -> list[list[str]]:
    """Group arms into rows for plotting.

    Row 1: root, trunk
    Row 2+: branch_N and its twigs (twig_1_bN, twig_2_bN, ...)

    Args:
        arm_names: List of arm names in order

    Returns:
        List of rows, each row is a list of arm names
    """
    from src.estimation.arm_types import classify_arm, ArmKind, get_branch_index

    rows: list[list[str]] = []

    # Row 1: root and trunk
    baseline_row = [n for n in arm_names if n in ("root", "trunk")]
    if baseline_row:
        rows.append(baseline_row)

    # Group branches with their twigs
    branches = [n for n in arm_names if classify_arm(n) == ArmKind.BRANCH]
    for branch in branches:
        branch_idx = get_branch_index(branch)
        row = [branch]
        # Find twigs for this branch
        for n in arm_names:
            if classify_arm(n) == ArmKind.TWIG:
                twig_branch_idx = get_branch_index(n)
                if twig_branch_idx == branch_idx:
                    row.append(n)
        rows.append(row)

    return rows


def plot_generalized_cores(
    result: "EstimationResult",
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create heatmap showing generalized cores for all arms organized in rows.

    Layout:
    - Row 1: root, trunk (baseline arms)
    - Row 2: branch_1 and its twigs
    - Row 3: branch_2 and its twigs
    - etc.

    Args:
        result: Estimation result containing arms with core_variants
        weighting_method: Which weighting method to use (e.g., "prob")
        structure_labels: Labels for structures (c1, c2, etc.)
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no data
    """
    # Get arms in proper order
    arm_names = [a.name for a in result.arms]
    ordered_names = get_ordered_arms_for_plotting(arm_names)
    arms_dict = {a.name: a for a in result.arms}

    if not ordered_names:
        return None

    # Group arms into rows
    rows = _group_arms_into_rows(ordered_names)
    if not rows:
        return None

    # Get variant data from first arm to determine structure
    first_arm = arms_dict.get(ordered_names[0])
    if not first_arm:
        return None
    first_variants = first_arm.estimates.get(weighting_method, {}).get("core_variants", [])
    if not first_variants:
        return None

    n_variants = len(first_variants)
    n_structures = len(structure_labels)
    n_rows = len(rows)
    max_cols = max(len(row) for row in rows)

    # Extract variant metadata for row labels
    variant_names = [v["name"] for v in first_variants]
    q_values = [v["q"] for v in first_variants]
    r_values = [v["r"] for v in first_variants]

    row_labels = []
    for name, q, r in zip(variant_names, q_values, r_values):
        q_str = _format_param(q)
        r_str = _format_param(r)
        row_labels.append(f"{name} (q={q_str}, r={r_str})")

    # Create figure with grid of subplots (extra width for colorbar)
    fig_width = max(20, n_structures * 1.8 * max_cols + 6)
    fig_height = max(12, n_variants * 0.6 * n_rows + 4)
    fig, axes = plt.subplots(n_rows, max_cols, figsize=(fig_width, fig_height),
                              squeeze=False)

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "diverging", ["#3B82F6", "#FFFFFF", "#EF4444"]
    )

    im = None  # For colorbar

    for row_idx, row_arms in enumerate(rows):
        for col_idx in range(max_cols):
            ax = axes[row_idx, col_idx]

            if col_idx >= len(row_arms):
                # Hide empty subplot
                ax.set_visible(False)
                continue

            arm_name = row_arms[col_idx]
            arm = arms_dict.get(arm_name)
            if not arm:
                ax.set_visible(False)
                continue

            estimates = arm.estimates.get(weighting_method, {})
            core_variants = estimates.get("core_variants", [])

            if not core_variants:
                ax.set_visible(False)
                continue

            # Build matrix: rows=variants, cols=structures
            matrix = np.zeros((n_variants, n_structures))
            for i, v in enumerate(core_variants):
                core = v.get("core", [])
                for j in range(min(len(core), n_structures)):
                    matrix[i, j] = core[j]

            # Plot heatmap
            im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

            # Set ticks
            ax.set_xticks(np.arange(n_structures))
            ax.set_xticklabels(structure_labels, fontsize=10)

            # Only show y-labels on first column of each row
            if col_idx == 0:
                ax.set_yticks(np.arange(n_variants))
                short_labels = [lbl[:25] + "..." if len(lbl) > 25 else lbl for lbl in row_labels]
                ax.set_yticklabels(short_labels, fontsize=9)
            else:
                ax.set_yticks([])

            # Title
            ax.set_title(arm_name.upper(), fontsize=12, fontweight="bold")

            # Add value annotations
            for i in range(n_variants):
                for j in range(n_structures):
                    val = matrix[i, j]
                    text_color = "white" if val < 0.35 or val > 0.65 else "black"
                    ax.text(
                        j, i, f"{val:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=9, fontweight="bold"
                    )

            # Grid lines
            ax.set_xticks(np.arange(n_structures + 1) - 0.5, minor=True)
            ax.set_yticks(np.arange(n_variants + 1) - 0.5, minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)

    # Suptitle
    fig.suptitle(
        f"Generalized Cores [{weighting_method}]",
        fontsize=14, fontweight="bold"
    )

    # Save
    plt.subplots_adjust(top=0.92, wspace=0.15, hspace=0.3)
    save_figure(plt.gcf(), output_path)
    return output_path


def plot_generalized_deviance(
    result: "EstimationResult",
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plots showing E[∂] as q→∞ (r=1) and r→∞ (q=1) for each arm.

    Layout organized by arm groups:
    - Row 1: root, trunk (q trajectory)
    - Row 2: root, trunk (r trajectory)
    - Row 3: branch_1 and twigs (q trajectory)
    - Row 4: branch_1 and twigs (r trajectory)
    - etc.

    Args:
        result: Estimation result containing arms with core_variants
        weighting_method: Which weighting method to use (e.g., "prob")
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no data
    """
    # Get arms in proper order
    arm_names = [a.name for a in result.arms]
    ordered_names = get_ordered_arms_for_plotting(arm_names)
    arms_dict = {a.name: a for a in result.arms}

    if not ordered_names:
        return None

    # Group arms into rows
    arm_rows = _group_arms_into_rows(ordered_names)
    if not arm_rows:
        return None

    n_arm_rows = len(arm_rows)
    max_cols = max(len(row) for row in arm_rows)

    # Create figure: 2 plot rows per arm row (q and r trajectories)
    fig_width = max(12, 4 * max_cols + 2)
    fig_height = max(8, 3 * n_arm_rows * 2 + 2)
    fig, axes = plt.subplots(n_arm_rows * 2, max_cols, figsize=(fig_width, fig_height),
                              squeeze=False)

    for arm_row_idx, row_arms in enumerate(arm_rows):
        for col_idx in range(max_cols):
            ax_q = axes[arm_row_idx * 2, col_idx]
            ax_r = axes[arm_row_idx * 2 + 1, col_idx]

            if col_idx >= len(row_arms):
                ax_q.set_visible(False)
                ax_r.set_visible(False)
                continue

            arm_name = row_arms[col_idx]
            arm = arms_dict.get(arm_name)
            if not arm:
                ax_q.set_visible(False)
                ax_r.set_visible(False)
                continue

            estimates = arm.estimates.get(weighting_method, {})
            core_variants = estimates.get("core_variants", [])

            if not core_variants:
                ax_q.set_visible(False)
                ax_r.set_visible(False)
                continue

            # Extract data
            data = []
            for v in core_variants:
                q = _parse_param(v["q"])
                r = _parse_param(v["r"])
                dev = v.get("deviance_avg", 0.0)
                name = v["name"]
                data.append({"q": q, "r": r, "dev": dev, "name": name})

            arm_color = get_arm_color(arm.name)

            # q trajectory (r=1 fixed)
            q_trajectory = [(d["q"], d["dev"], d["name"]) for d in data if d["r"] == 1.0]
            _plot_single_trajectory(ax_q, q_trajectory, "q", arm_color)
            ax_q.set_title(f"{arm.name.upper()}", fontsize=10, fontweight="bold")

            # r trajectory (q=1 fixed)
            r_trajectory = [(d["r"], d["dev"], d["name"]) for d in data if d["q"] == 1.0]
            _plot_single_trajectory(ax_r, r_trajectory, "r", arm_color)

            # Y-axis labels on first column only
            if col_idx == 0:
                ax_q.set_ylabel("E[∂] (q)", fontsize=9)
                ax_r.set_ylabel("E[∂] (r)", fontsize=9)

    # Suptitle - position lower to avoid overlap with subplot titles
    fig.suptitle(
        f"Generalized Deviance [{weighting_method}]",
        fontsize=13, fontweight="bold", y=0.995
    )

    # Save - leave more top margin
    plt.subplots_adjust(left=0.08, top=0.96, hspace=0.5, wspace=0.2)
    save_figure(plt.gcf(), output_path)
    return output_path


def _plot_single_trajectory(
    ax,
    trajectory: list[tuple[float, float, str]],
    param_name: str,
    color: str,
) -> None:
    """Plot E[∂] along a single parameter trajectory.

    Args:
        ax: Matplotlib axis
        trajectory: List of (param_value, deviance, variant_name)
        param_name: "q" or "r"
        color: Line color
    """
    if not trajectory:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    # Sort by parameter value (handle infinities)
    def sort_key(x):
        val = x[0]
        if val == float("-inf"):
            return (-2, 0)
        elif val == float("inf"):
            return (2, 0)
        else:
            return (0, val)

    sorted_traj = sorted(trajectory, key=sort_key)

    # Create x positions and labels
    x_positions = []
    x_labels = []
    y_vals = []
    point_labels = []

    for i, (val, dev, name) in enumerate(sorted_traj):
        x_positions.append(i)
        x_labels.append(_format_param(val))
        y_vals.append(dev)
        point_labels.append(name)

    # Plot line and markers
    ax.plot(x_positions, y_vals, "o-", color=color, linewidth=2.5,
            markersize=8, markeredgecolor="white", markeredgewidth=1.5)

    # Only add labels for extreme points (first and last) to reduce clutter
    if len(x_positions) > 0:
        # Label only the first and last points to avoid clutter
        for idx in [0, len(x_positions) - 1] if len(x_positions) > 1 else [0]:
            x, y, name = x_positions[idx], y_vals[idx], point_labels[idx]
            # Shorter name (first word only)
            short_name = name.split("_")[0] if "_" in name else name[:8]
            ax.annotate(
                short_name, (x, y),
                xytext=(0, 8), textcoords="offset points",
                fontsize=8, ha="center", va="bottom",
                color="#555", alpha=0.8,
            )

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel(param_name, fontsize=12)

    # Styling
    style_axis_clean(ax, grid_axis="both")
    ax.tick_params(axis="y", labelsize=10)

    # Y limits with padding
    if y_vals:
        y_min, y_max = min(y_vals), max(y_vals)
        padding = (y_max - y_min) * 0.15 if y_max > y_min else 0.1
        ax.set_ylim(y_min - padding, y_max + padding * 2)


def _param_to_plot_val(val: float) -> float:
    """Convert parameter value to plottable x-coordinate."""
    if val == float("inf"):
        return 10.0  # Represent infinity as 10
    elif val == float("-inf"):
        return -10.0  # Represent -infinity as -10
    return val


def _parse_param(val: float | str) -> float:
    """Parse q or r parameter value from JSON."""
    if isinstance(val, str):
        if val.lower() in ("inf", "infinity"):
            return float("inf")
        elif val.lower() in ("-inf", "-infinity"):
            return float("-inf")
        return float(val)
    return val


def _format_param(val: float | str) -> str:
    """Format q or r parameter value."""
    if isinstance(val, str):
        if val.lower() in ("inf", "infinity"):
            return "∞"
        elif val.lower() in ("-inf", "-infinity"):
            return "-∞"
        return val

    if val == float("inf"):
        return "∞"
    elif val == float("-inf"):
        return "-∞"
    elif val == int(val):
        return str(int(val))
    else:
        return f"{val:.1f}"
