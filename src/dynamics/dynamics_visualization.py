"""Visualization for dynamics analysis.

Plots three metrics over token position k:
- Pull: l2 norm of scores (normative strength)
- Drift: deviance from initial scores (how far we've moved from start)
- Potential: deviance from final scores (how far to end state)

Output structure:
    dynamics/
        all/                         # Individual trajectory plots
            traj_0_trunk.png
            traj_1_branch_1.png
        dynamics_trunk.png           # Aggregate: all trunk trajectories overlaid
        dynamics_branch.png          # Aggregate: all branch trajectories overlaid
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt

from src.viz.viz_style_config import (
    AXIS_LABEL_FONTSIZE,
    DPI,
    DYNAMICS_LINE_WIDTH,
    DYNAMICS_MARKER_EDGE_WIDTH,
    DYNAMICS_MARKER_SIZE,
    FACECOLOR,
    FIGURE_SIZE_DYNAMICS,
    GRID_ALPHA_MAJOR,
    GRID_ALPHA_MINOR,
    GRID_LINE_WIDTH_MAJOR,
    GRID_LINE_WIDTH_MINOR,
    LEGEND_FONTSIZE,
    TICK_LABEL_FONTSIZE,
    TITLE_FONTSIZE,
    TITLE_FONTWEIGHT,
    get_dynamics_color,
)

from .dynamics_types import DynamicsResult, TrajectoryDynamics


def _style_axis(ax: plt.Axes, title: str, ylabel: str = "Value") -> None:
    """Apply consistent styling to an axis."""
    ax.set_facecolor(FACECOLOR)
    ax.set_xlabel("Token Position (k)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE, fontweight=TITLE_FONTWEIGHT)
    ax.grid(True, which="major", alpha=GRID_ALPHA_MAJOR, linewidth=GRID_LINE_WIDTH_MAJOR, zorder=1)
    ax.grid(True, which="minor", alpha=GRID_ALPHA_MINOR, linewidth=GRID_LINE_WIDTH_MINOR, zorder=1)
    ax.minorticks_on()
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    ax.set_ylim(bottom=0)


def _plot_series(ax: plt.Axes, data: list[tuple[int, float]], color: str, label: str, marker: str) -> None:
    """Plot a single series with line and markers."""
    if not data:
        return
    positions = [k for k, _ in data]
    values = [v for _, v in data]
    ax.plot(positions, values, color=color, linewidth=DYNAMICS_LINE_WIDTH, label=label, zorder=3)
    ax.scatter(
        positions, values, marker=marker, s=DYNAMICS_MARKER_SIZE**2,
        facecolors="white", edgecolors=color, linewidths=DYNAMICS_MARKER_EDGE_WIDTH, zorder=4
    )


def _plot_single_trajectory(traj: TrajectoryDynamics, ax: plt.Axes) -> None:
    """Plot pull, drift, and potential curves for a single trajectory."""
    if not traj.positions:
        return

    _plot_series(ax, traj.pull_series, get_dynamics_color("pull"), "Pull", "^")
    _plot_series(ax, traj.drift_series, get_dynamics_color("drift"), "Drift", "o")
    _plot_series(ax, traj.potential_series, get_dynamics_color("potential"), "Potential", "s")

    _style_axis(ax, f"Trajectory {traj.traj_idx} ({traj.arm_name})")
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")


def _plot_aggregate_arm(trajs: list[TrajectoryDynamics], arm_type: str) -> plt.Figure:
    """Create 3-column aggregate plot for an arm type (pull, drift, potential)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=FACECOLOR)

    metrics = [
        ("Pull", "pull", "^", lambda t: t.pull_series),
        ("Drift", "drift", "o", lambda t: t.drift_series),
        ("Potential", "potential", "s", lambda t: t.potential_series),
    ]

    for ax, (name, key, marker, get_series) in zip(axes, metrics):
        color = get_dynamics_color(key)
        for i, traj in enumerate(trajs):
            data = get_series(traj)
            if not data:
                continue
            positions = [k for k, _ in data]
            values = [v for _, v in data]
            alpha = 0.7
            label = f"traj {traj.traj_idx}" if i < 5 else None  # Limit legend entries
            ax.plot(positions, values, color=color, linewidth=DYNAMICS_LINE_WIDTH, alpha=alpha, label=label, zorder=3)
            ax.scatter(
                positions, values, marker=marker, s=(DYNAMICS_MARKER_SIZE * 0.8)**2,
                facecolors="white", edgecolors=color, linewidths=DYNAMICS_MARKER_EDGE_WIDTH, alpha=alpha, zorder=4
            )

        _style_axis(ax, name, name)
        if len(trajs) <= 5:
            ax.legend(fontsize=LEGEND_FONTSIZE - 2, loc="upper right")

    fig.suptitle(f"Dynamics: {arm_type}", fontsize=TITLE_FONTSIZE + 2, fontweight=TITLE_FONTWEIGHT)
    plt.tight_layout()
    return fig


def plot_dynamics(result: DynamicsResult, output_dir: Path) -> list[Path]:
    """Generate dynamics plots.

    Output:
        {output_dir}/all/traj_{idx}_{arm}.png  - Individual trajectory plots
        {output_dir}/dynamics_{arm_type}.png   - Aggregate per arm type
    """
    all_dir = output_dir / "all"
    all_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    # Individual plots in all/
    for traj in result.trajectories:
        if not traj.positions:
            continue
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_DYNAMICS, facecolor=FACECOLOR)
        _plot_single_trajectory(traj, ax)
        plt.tight_layout()
        path = all_dir / f"traj_{traj.traj_idx}_{traj.arm_name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
        saved_paths.append(path)

    # Aggregate plots per arm (branch_1, branch_2, etc.)
    trajs_with_positions = [t for t in result.trajectories if t.positions]
    sorted_trajs = sorted(trajs_with_positions, key=lambda t: t.arm_name)

    for arm_name, group in groupby(sorted_trajs, key=lambda t: t.arm_name):
        trajs = list(group)
        if not trajs:
            continue
        fig = _plot_aggregate_arm(trajs, arm_name)
        path = output_dir / f"dynamics_{arm_name}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=FACECOLOR)
        plt.close(fig)
        saved_paths.append(path)

    return saved_paths
