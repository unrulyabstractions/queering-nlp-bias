"""Deviance and orientation visualization.

Shows deviance changes across arms and orientation vectors per branch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

BRANCH_COLORS = ["#4A90D9", "#E67E22", "#2ECC71", "#9B59B6", "#E74C3C"]


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

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.array([0, 1])

    for i, arm in enumerate(branch_arms):
        branch_dev = arm.get_deviance_avg(weighting_method)
        delta = branch_dev - trunk_dev
        color = BRANCH_COLORS[i % len(BRANCH_COLORS)]

        ax.plot(x, [trunk_dev, branch_dev], color=color, marker="o",
                markersize=10, linewidth=2.5, label=arm.name, alpha=0.9)

        if i == 0:
            ax.annotate(f"E[∂|T] = {trunk_dev:.4f}", (0, trunk_dev),
                        xytext=(-10, 10), textcoords="offset points",
                        ha="right", va="bottom", fontsize=9, fontweight="medium")

        sign = "+" if delta >= 0 else ""
        ax.annotate(f"E[∂|B] = {branch_dev:.4f}\nE[Δ∂] = {sign}{delta:.4f}",
                    (1, branch_dev), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9, fontweight="medium", color=color)

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
