"""Visualization for dynamics analysis.

Drift y(k): line showing deviance from root core as text develops
Horizon z(k): line showing deviance from each arm's core along trajectory path
Pull x(k): line showing l2 norm of each arm's core along trajectory path
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .dynamics_types import DynamicsResult, TrajectoryDynamics

# Fixed colors for curves (no arm-specific colors)
DRIFT_COLOR = "#8E44AD"  # Purple
HORIZON_COLOR = "#2980B9"  # Blue
PULL_COLOR = "#E67E22"  # Orange


def _plot_single_trajectory(
    traj: TrajectoryDynamics,
    ax: plt.Axes,
) -> None:
    """Plot drift, horizon, and pull curves for a single trajectory.

    All curves use the same x-axis: total token position from start of generation.
    - Horizon/Pull are plotted at arm prefix positions (where each arm's conditioning ends)
    - Drift is shifted by the trajectory's arm prefix length (continuation starts after prefix)

    Drift: deviance(partial_text, root_core) over token position
    Horizon: deviance(full_text, arm_core) at each arm's prefix position (connected)
    Pull: l2 norm of arm's core at each arm's prefix position (connected)
    """
    # Get the prefix length for this trajectory's arm (to shift drift)
    traj_prefix_tokens = 0
    if traj.pull_points:
        # Last pull point is for this trajectory's own arm
        traj_prefix_tokens = traj.pull_points[-1].arm_prefix_tokens

    # === DRIFT LINE ===
    # Shift drift positions by prefix length so x-axis = total tokens
    if traj.drift_points:
        positions = [traj_prefix_tokens + p.token_position for p in traj.drift_points]
        deviances = [p.deviance for p in traj.drift_points]

        ax.plot(
            positions,
            deviances,
            color=DRIFT_COLOR,
            linewidth=2,
            label="Drift",
            marker="o",
            markersize=4,
            markerfacecolor="white",
            markeredgecolor=DRIFT_COLOR,
            markeredgewidth=1,
        )

    # === HORIZON LINE ===
    if traj.horizon_points:
        h_positions = [hp.arm_prefix_tokens for hp in traj.horizon_points]
        h_deviances = [hp.deviance for hp in traj.horizon_points]

        ax.plot(
            h_positions,
            h_deviances,
            color=HORIZON_COLOR,
            linewidth=2,
            label="Horizon",
            marker="s",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=HORIZON_COLOR,
            markeredgewidth=1,
        )

    # === PULL LINE ===
    if traj.pull_points:
        p_positions = [pp.arm_prefix_tokens for pp in traj.pull_points]
        p_values = [pp.pull for pp in traj.pull_points]

        ax.plot(
            p_positions,
            p_values,
            color=PULL_COLOR,
            linewidth=2,
            label="Pull",
            marker="^",
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=PULL_COLOR,
            markeredgewidth=1,
        )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Value")
    ax.set_title(f"Trajectory {traj.traj_idx} ({traj.arm_name})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def plot_dynamics(
    result: DynamicsResult,
    output_dir: Path,
) -> list[Path]:
    """Generate dynamics plots for all trajectories.

    Output: one PNG per trajectory in a flat folder structure.
    - {output_dir}/traj_{idx}_{arm_name}.png
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for traj in result.trajectories:
        if not traj.drift_points and not traj.horizon_points:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_single_trajectory(traj, ax)

        path = output_dir / f"traj_{traj.traj_idx}_{traj.arm_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
