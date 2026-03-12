"""Logging utilities for dynamics computation."""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.logging import log

from ..dynamics_types import DynamicsResult, TrajectoryDynamics


def _log_step(step_num: int, title: str, log_fn: LogFn) -> None:
    log_fn(f"\n=={step_num}== {title}")
    log_fn("  " + "─" * 50)


def _log_divider(width: int, log_fn: LogFn) -> None:
    log_fn("─" * width)


def log_dynamics_header(
    n_trajectories: int,
    n_arms: int,
    drift_step: int,
    log_fn: LogFn | None = None,
) -> None:
    """Log dynamics computation header."""
    _log = log_fn or log
    _log_step(0, "Dynamics Computation", _log)
    _log(f"  Trajectories: {n_trajectories}")
    _log(f"  Arms: {n_arms}")
    _log(f"  Drift step: every {drift_step} tokens")
    _log(f"  Horizon/Pull points per traj: {n_arms}")
    _log_divider(60, _log)


def log_trajectory_start(
    traj_idx: int,
    arm_name: str,
    n_tokens: int,
    n_drift_points: int,
    n_horizon_points: int,
    log_fn: LogFn | None = None,
) -> None:
    """Log start of trajectory dynamics computation."""
    _log = log_fn or log
    _log(f"\n  Trajectory [{traj_idx}] ({arm_name}, {n_tokens} tokens)")
    _log(f"    {n_drift_points} drift + {n_horizon_points} horizon")


def log_drift_point(
    position: int,
    n_tokens: int,
    scores: list[float],
    deviance: float,
    log_fn: LogFn | None = None,
) -> None:
    """Log a drift measurement (partial text scored, compared to root)."""
    _log = log_fn or log
    pct = (position / n_tokens * 100) if n_tokens > 0 else 0
    scores_str = ", ".join(f"{s:.2f}" for s in scores)
    _log(f"    drift @{position:3d} ({pct:4.0f}%): scores=[{scores_str}] -> dev={deviance:.3f}")


def log_horizon_point(
    arm_name: str,
    prefix_tokens: int,
    deviance: float,
    is_own_arm: bool,
    log_fn: LogFn | None = None,
) -> None:
    """Log a horizon measurement (full text vs arm core)."""
    _log = log_fn or log
    marker = " *" if is_own_arm else ""
    _log(f"    horizon({arm_name}, @{prefix_tokens}): dev={deviance:.3f}{marker}")


def log_trajectory_summary(
    traj: TrajectoryDynamics,
    log_fn: LogFn | None = None,
) -> None:
    """Log summary of trajectory dynamics."""
    _log = log_fn or log
    n_drift = len(traj.drift_points)
    n_horizon = len(traj.horizon_points)

    # Find own-arm horizon
    own_horizon = next(
        (h.deviance for h in traj.horizon_points if h.arm_name == traj.arm_name),
        0.0,
    )
    # Find closest arm
    if traj.horizon_points:
        closest = min(traj.horizon_points, key=lambda h: h.deviance)
        _log(f"    Summary: {n_drift} drift, {n_horizon} horizon | own={own_horizon:.3f}, closest={closest.arm_name}({closest.deviance:.3f})")
    else:
        _log(f"    Summary: {n_drift} drift, {n_horizon} horizon")


def log_pull_point(
    arm_name: str,
    prefix_tokens: int,
    pull: float,
    log_fn: LogFn | None = None,
) -> None:
    """Log a pull measurement (l2 norm of arm core)."""
    _log = log_fn or log
    _log(f"    pull({arm_name}, @{prefix_tokens}): ||core||={pull:.3f}")


def log_dynamics_result(
    result: DynamicsResult,
    log_fn: LogFn | None = None,
) -> None:
    """Log final dynamics result summary."""
    _log = log_fn or log

    _log_divider(60, _log)
    _log_step(1, "Dynamics Complete", _log)

    total_drift = sum(len(t.drift_points) for t in result.trajectories)
    total_horizon = sum(len(t.horizon_points) for t in result.trajectories)
    total_pull = sum(len(t.pull_points) for t in result.trajectories)

    _log(f"  Trajectories: {len(result.trajectories)}")
    _log(f"  Total drift points: {total_drift}")
    _log(f"  Total horizon points: {total_horizon}")
    _log(f"  Total pull points: {total_pull}")
    _log(f"  Arm prefix tokens: {result.arm_prefix_tokens}")
