"""Generalized cores and deviance visualizations.

Shows different statistical variants (standard, uniform, mode, etc.) with their
(q, r) parameters as heatmaps and line plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

# Consistent arm colors
ARM_COLORS = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C"]


def plot_generalized_cores(
    result: "EstimationResult",
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create heatmap showing generalized cores for all arms side by side.

    Each arm gets a column of heatmaps showing core values across variants.
    Rows are variant names (standard, uniform, mode, etc.)
    Columns within each arm are structure labels (c1, c2, etc.)

    Args:
        result: Estimation result containing arms with core_variants
        weighting_method: Which weighting method to use (e.g., "prob")
        structure_labels: Labels for structures (c1, c2, etc.)
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no data
    """
    # Get arms (skip all_arms)
    arms = [a for a in result.arms if a.name != "all_arms"]
    if not arms:
        return None

    # Get variant data from first arm to determine structure
    first_variants = arms[0].estimates.get(weighting_method, {}).get("core_variants", [])
    if not first_variants:
        return None

    n_variants = len(first_variants)
    n_structures = len(structure_labels)
    n_arms = len(arms)

    # Extract variant metadata
    variant_names = [v["name"] for v in first_variants]
    q_values = [v["q"] for v in first_variants]
    r_values = [v["r"] for v in first_variants]

    # Row labels with q, r parameters
    row_labels = []
    for name, q, r in zip(variant_names, q_values, r_values):
        q_str = _format_param(q)
        r_str = _format_param(r)
        row_labels.append(f"{name} (q={q_str}, r={r_str})")

    # Create figure with subplots for each arm - much larger for readability
    fig_width = max(22, n_structures * 2.0 * n_arms + 5)
    fig_height = max(18, n_variants * 1.0 + 5)
    fig, axes = plt.subplots(1, n_arms, figsize=(fig_width, fig_height), sharey=True)
    if n_arms == 1:
        axes = [axes]

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "diverging", ["#3B82F6", "#FFFFFF", "#EF4444"]
    )

    for arm_idx, (arm, ax) in enumerate(zip(arms, axes)):
        estimates = arm.estimates.get(weighting_method, {})
        core_variants = estimates.get("core_variants", [])

        if not core_variants:
            continue

        # Build matrix: rows=variants, cols=structures (no E[∂])
        matrix = np.zeros((n_variants, n_structures))
        for i, v in enumerate(core_variants):
            core = v.get("core", [])
            for j in range(min(len(core), n_structures)):
                matrix[i, j] = core[j]

        # Plot heatmap
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(n_structures))
        ax.set_xticklabels(structure_labels, fontsize=13)
        if arm_idx == 0:
            ax.set_yticks(np.arange(n_variants))
            # Larger row labels, truncate long names
            short_labels = [lbl[:30] + "..." if len(lbl) > 30 else lbl for lbl in row_labels]
            ax.set_yticklabels(short_labels, fontsize=12)

        # Title
        ax.set_title(arm.name.upper(), fontsize=16, fontweight="bold")

        # Add value annotations with better contrast and larger font
        for i in range(n_variants):
            for j in range(n_structures):
                val = matrix[i, j]
                # Use white text on dark backgrounds (both red and blue extremes)
                text_color = "white" if val < 0.35 or val > 0.65 else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold"
                )

        # Grid lines
        ax.set_xticks(np.arange(n_structures + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_variants + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Core Value", fontsize=10)

    # Suptitle
    fig.suptitle(
        f"Generalized Cores [{weighting_method}]",
        fontsize=13, fontweight="bold"
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(top=0.92, wspace=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def plot_generalized_deviance(
    result: "EstimationResult",
    weighting_method: str,
    output_path: Path,
) -> Path | None:
    """Create line plots showing E[∂] as q→∞ (r=1) and r→∞ (q=1) for each arm.

    Layout: columns = arms, rows = [q trajectory, r trajectory]
    Each subplot shows a single clean line from -∞ to +∞.

    Args:
        result: Estimation result containing arms with core_variants
        weighting_method: Which weighting method to use (e.g., "prob")
        output_path: Where to save the plot

    Returns:
        Path to saved file, or None if no data
    """
    # Get arms (skip all_arms)
    arms = [a for a in result.arms if a.name != "all_arms"]
    if not arms:
        return None

    n_arms = len(arms)

    # Create figure: 2 rows x n_arms columns - larger for readability
    fig_width = max(14, 6 * n_arms)
    fig_height = max(10, 12)
    fig, axes = plt.subplots(2, n_arms, figsize=(fig_width, fig_height), squeeze=False, sharey="row")

    for arm_idx, arm in enumerate(arms):
        estimates = arm.estimates.get(weighting_method, {})
        core_variants = estimates.get("core_variants", [])

        if not core_variants:
            continue

        # Extract data
        data = []
        for v in core_variants:
            q = _parse_param(v["q"])
            r = _parse_param(v["r"])
            dev = v.get("deviance_avg", 0.0)
            name = v["name"]
            data.append({"q": q, "r": r, "dev": dev, "name": name})

        # --- Top row: q trajectory (r=1 fixed) ---
        ax_q = axes[0, arm_idx]
        q_trajectory = [(d["q"], d["dev"], d["name"]) for d in data if d["r"] == 1.0]
        _plot_single_trajectory(ax_q, q_trajectory, "q", ARM_COLORS[arm_idx % len(ARM_COLORS)])
        ax_q.set_title(f"{arm.name.upper()}", fontsize=11, fontweight="bold")

        # --- Bottom row: r trajectory (q=1 fixed) ---
        ax_r = axes[1, arm_idx]
        r_trajectory = [(d["r"], d["dev"], d["name"]) for d in data if d["q"] == 1.0]
        _plot_single_trajectory(ax_r, r_trajectory, "r", ARM_COLORS[arm_idx % len(ARM_COLORS)])

    # Y-axis labels
    axes[0, 0].set_ylabel("E[∂]", fontsize=12)
    axes[1, 0].set_ylabel("E[∂]", fontsize=12)

    # Row labels - darker color for better visibility
    fig.text(0.02, 0.72, "q trajectory\n(r = 1)", fontsize=12, fontweight="bold",
             ha="center", va="center", rotation=90, color="#333")
    fig.text(0.02, 0.28, "r trajectory\n(q = 1)", fontsize=12, fontweight="bold",
             ha="center", va="center", rotation=90, color="#333")

    # Suptitle
    fig.suptitle(
        f"Generalized Deviance [{weighting_method}]",
        fontsize=13, fontweight="bold"
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.subplots_adjust(left=0.12, top=0.90, hspace=0.35, wspace=0.15)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

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

    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
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
