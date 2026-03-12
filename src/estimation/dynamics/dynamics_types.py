"""Data types for dynamics analysis.

Drift y(k): deviance of PARTIAL text (at token k) relative to root core
Horizon z(arm): deviance of FULL text relative to arm's core, plotted at arm's prefix length
Pull x(arm): l2 norm of arm's core, plotted at arm's prefix length
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema


@dataclass
class DriftPoint(BaseSchema):
    """Drift measurement at a specific token position.

    Drift = deviance(partial_text_scores, root_core)
    """
    token_position: int
    partial_scores: list[float]  # Scores from re-scoring partial text
    deviance: float  # Deviance from root core


@dataclass
class HorizonPoint(BaseSchema):
    """Horizon measurement at an arm position.

    Horizon = deviance(full_traj_scores, arm_core)
    Plotted at the arm's prefix token count.
    """
    arm_name: str
    arm_prefix_tokens: int  # X-axis position (token count of arm prefix)
    deviance: float  # Deviance from this arm's core


@dataclass
class PullPoint(BaseSchema):
    """Pull measurement at an arm position.

    Pull = l2 norm of arm's core vector.
    Represents the "strength" of the arm's normative characterization.
    """
    arm_name: str
    arm_prefix_tokens: int  # X-axis position
    pull: float  # l2 norm of core


@dataclass
class TrajectoryDynamics(BaseSchema):
    """Dynamics data for a single trajectory."""

    traj_idx: int
    arm_name: str
    full_text: str
    n_tokens: int
    full_scores: list[float]  # Scores of full trajectory (pre-computed)

    # Drift: re-scored partial text at various k, compared to root core
    drift_points: list[DriftPoint] = field(default_factory=list)

    # Horizon: full traj scores compared to each arm's core
    horizon_points: list[HorizonPoint] = field(default_factory=list)

    # Pull: l2 norm of each arm's core
    pull_points: list[PullPoint] = field(default_factory=list)


@dataclass
class DynamicsResult(BaseSchema):
    """Result of dynamics computation for all trajectories."""

    root_core: list[float]
    arm_cores: dict[str, list[float]]  # arm_name -> core
    arm_prefix_tokens: dict[str, int]  # arm_name -> prefix token count
    trajectories: list[TrajectoryDynamics] = field(default_factory=list)
