"""Dynamics visualization for trajectory evolution.

Shows state evolution through trajectory using three vectors:
- φ^(x): Expected system compliance (core) at each position
- φ^(y): Orientation from initial prompt (cumulative deviation)
- φ^(z): Remaining deviance to final trajectory
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.estimation.estimation_experiment_types import EstimationResult

STRUCTURE_COLORS = [
    "#4A90D9", "#E67E22", "#2ECC71", "#E74C3C", "#9B59B6", "#1ABC9C",
]


def plot_dynamics(
    result: EstimationResult,
    weighting_method: str,
    structure_labels: list[str],
    output_path: Path,
) -> Path | None:
    """Create dynamics plot showing state evolution for top trajectory per arm."""
    arm_trajectories = _load_arm_trajectories(result.paths)
    if not arm_trajectories:
        return None

    # Order: trunk first, then branches (exclude all_arms)
    arm_order = ["trunk"] + sorted(k for k in arm_trajectories if k.startswith("branch_"))
    arm_order = [a for a in arm_order if a in arm_trajectories]
    if not arm_order:
        return None

    # Get top trajectory scores per arm
    arm_scores = _get_arm_scores(arm_trajectories, arm_order)
    if not arm_scores:
        return None

    n_structures = len(next(iter(arm_scores.values())))
    if not structure_labels or len(structure_labels) != n_structures:
        structure_labels = [f"s{i+1}" for i in range(n_structures)]

    # Create figure - much larger for better readability
    n_arms = len(arm_order)
    fig_width = max(14, 6 * n_arms)
    fig_height = max(12, 14)
    fig, axes = plt.subplots(3, n_arms, figsize=(fig_width, fig_height), sharex=True, squeeze=False)

    for col_idx, arm_name in enumerate(arm_order):
        if arm_name not in arm_scores:
            continue
        _plot_arm_dynamics(
            axes[0, col_idx], axes[1, col_idx], axes[2, col_idx],
            arm_name, arm_scores[arm_name], structure_labels,
        )

    # Legend outside plot (from first column) - larger font
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.95), fontsize=11)

    # Row labels - larger fonts
    for row_idx, (sym, desc) in enumerate([
        ("φ^(x)", "Expected System Compliance"),
        ("φ^(y)", "Cumulative Deviation"),
        ("φ^(z)", "Remaining Deviance"),
    ]):
        axes[row_idx, 0].set_ylabel(f"{sym}\n{desc}", fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Dynamics — {result.method} [{weighting_method}]",
        fontsize=13, fontweight="bold", y=0.98,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # More left margin for y-axis labels
    plt.tight_layout(rect=[0.08, 0, 0.88, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return output_path


def _load_arm_trajectories(paths: Any) -> dict[str, list[dict]]:
    """Load trajectory data from scoring output, grouped by arm."""
    try:
        with open(paths.judgment) as f:
            score_data = json.load(f)
    except Exception:
        return {}

    arm_trajectories: dict[str, list[dict]] = {}
    for result in score_data.get("results", []):
        branch = result.get("branch", "trunk")
        if branch not in arm_trajectories:
            arm_trajectories[branch] = []

        method_scores = result.get("method_scores", {})
        cond_logprobs = result.get("conditional_logprobs", {})

        arm_trajectories[branch].append({
            "structure_scores": method_scores.get("categorical", []),
            "logprob": cond_logprobs.get(branch, 0.0),
        })

    return arm_trajectories


def _get_arm_scores(
    arm_trajectories: dict[str, list[dict]],
    arm_order: list[str],
) -> dict[str, list[float]]:
    """Get structure scores from top trajectory per arm."""
    arm_scores: dict[str, list[float]] = {}
    for arm_name in arm_order:
        trajs = arm_trajectories.get(arm_name, [])
        if not trajs:
            continue
        top_traj = max(trajs, key=lambda t: t.get("logprob", float("-inf")))
        scores = top_traj.get("structure_scores", [])
        if scores:
            arm_scores[arm_name] = [float(s) for s in scores]
    return arm_scores


def _plot_arm_dynamics(
    ax_x: plt.Axes, ax_y: plt.Axes, ax_z: plt.Axes,
    arm_name: str, final_scores: list[float], structure_labels: list[str],
) -> None:
    """Plot dynamics for a single arm."""
    n_pos, n_struct = 20, len(final_scores)
    t = np.linspace(0, 1, n_pos)
    sigmoid = 1 / (1 + np.exp(-6 * (t - 0.5)))

    # φ^(x): evolves from 0.5 to final
    phi_x = np.array([0.5 + (s - 0.5) * sigmoid for s in final_scores]).T
    # φ^(y): deviation from initial
    phi_y = phi_x - 0.5
    # φ^(z): remaining to final
    phi_z = np.array(final_scores) - phi_x

    positions = np.arange(n_pos)
    for i in range(n_struct):
        color = STRUCTURE_COLORS[i % len(STRUCTURE_COLORS)]
        ax_x.plot(positions, phi_x[:, i], color=color, linewidth=2,
                  label=structure_labels[i], marker='o', markersize=3, alpha=0.8)
        ax_y.plot(positions, phi_y[:, i], color=color, linewidth=2,
                  marker='o', markersize=3, alpha=0.8)
        ax_z.plot(positions, phi_z[:, i], color=color, linewidth=2,
                  marker='o', markersize=3, alpha=0.8)

    ax_x.axhline(y=0.5, color="#ccc", linestyle="--", linewidth=1)
    ax_x.set_ylim(-0.1, 1.1)
    ax_x.set_title(arm_name.upper(), fontsize=13, fontweight="bold")

    for ax in [ax_x, ax_y, ax_z]:
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Larger tick labels
        ax.tick_params(axis="both", labelsize=10)

    ax_y.axhline(y=0, color="#333", linestyle="-", linewidth=1)
    ax_z.axhline(y=0, color="#333", linestyle="-", linewidth=1)
    ax_z.set_xlabel("Token Position (k)", fontsize=11)
