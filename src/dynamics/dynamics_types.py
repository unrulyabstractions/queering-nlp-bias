"""Data types for dynamics analysis.

At each token position k, we:
1. Score the partial text to get a "core estimate" (structure scores)
2. Compute metrics from those scores:
   - Pull: l2 norm of scores (normative strength)
   - Drift: deviance from initial scores (how far we've moved)
   - Horizon: deviance from final scores (how far to end state)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema


@dataclass
class PositionScores(BaseSchema):
    """Scores computed at a specific token position."""

    k: int  # Token position
    scores: list[float]  # Structure scores at this position
    pull: float  # l2 norm of scores
    drift: float  # Deviance from initial scores
    horizon: float  # Deviance from final scores


@dataclass
class TrajectoryDynamics(BaseSchema):
    """Dynamics for a single trajectory."""

    traj_idx: int
    arm_name: str
    text: str
    n_tokens: int
    positions: list[PositionScores]  # Scores at each measured position

    @property
    def pull_series(self) -> list[tuple[int, float]]:
        """(k, pull) pairs for plotting."""
        return [(p.k, p.pull) for p in self.positions]

    @property
    def drift_series(self) -> list[tuple[int, float]]:
        """(k, drift) pairs for plotting."""
        return [(p.k, p.drift) for p in self.positions]

    @property
    def horizon_series(self) -> list[tuple[int, float]]:
        """(k, horizon) pairs for plotting."""
        return [(p.k, p.horizon) for p in self.positions]


@dataclass
class DynamicsResult(BaseSchema):
    """Result of dynamics computation."""

    trajectories: list[TrajectoryDynamics] = field(default_factory=list)
    n_structures: int = 0  # Number of structure dimensions
    step: int = 2  # Token step between measurements
