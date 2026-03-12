"""Logging utilities for scoring methods.

This module provides logging functions for detailed scoring output.
"""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.logging import oneline
from src.common.viz_utils import preview
from src.scoring.scoring_data import TrajectoryData


def log_trajectory_header(
    traj: TrajectoryData,
    idx: int,
    total: int,
    log_fn: LogFn,
) -> None:
    """Log trajectory header with response text.

    Args:
        traj: Trajectory being scored
        idx: Current trajectory index (0-based)
        total: Total number of trajectories
        log_fn: Logging callback
    """
    branch_display = "trunk" if traj.branch_idx == 0 else f"branch_{traj.branch_idx}"
    log_fn(f"Trajectory {idx + 1}/{total} (branch: {branch_display})")
    log_fn(f'  Response: "{preview(oneline(traj.response), 120)}"')


def log_scoring_section(section_name: str, log_fn: LogFn) -> None:
    """Log a scoring section header.

    Args:
        section_name: Name of the section (e.g., "Categorical", "Graded", "Similarity")
        log_fn: Logging callback
    """
    log_fn(f"  {section_name}:")
