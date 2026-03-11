"""Auxiliary data types for estimation.

This module defines helper data structures for organizing and summarizing
estimation results, including trajectory continuations and arm summaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

# ══════════════════════════════════════════════════════════════════════════════
# CONTINUATION TYPES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryContinuation(BaseSchema):
    """A trajectory continuation with its index."""

    traj_idx: int
    text: str


@dataclass
class ContinuationsByArm(BaseSchema):
    """Trajectory continuations organized by arm name."""

    by_arm: dict[str, list[TrajectoryContinuation]] = field(default_factory=dict)

    def get(self, arm_name: str) -> list[TrajectoryContinuation]:
        """Get continuations for an arm."""
        return self.by_arm.get(arm_name, [])

    def add(self, arm_name: str, traj_idx: int, text: str) -> None:
        """Add a continuation to an arm."""
        if arm_name not in self.by_arm:
            self.by_arm[arm_name] = []
        self.by_arm[arm_name].append(
            TrajectoryContinuation(traj_idx=traj_idx, text=text)
        )

    def arms(self) -> list[str]:
        """Get list of arm names."""
        return list(self.by_arm.keys())

    def __iter__(self):
        """Allow iteration over (arm_name, continuations) pairs."""
        return iter(self.by_arm.items())

    def items(self):
        """Return (arm_name, continuations) pairs."""
        return self.by_arm.items()


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TYPES
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryGrouping(BaseSchema):
    """Maps a trajectory to its arm membership(s) and text."""

    traj_idx: int
    arm_idxs: list[int]  # Trajectory can belong to multiple arms (trunk + its branch)
    continuation_text: str


@dataclass
class ArmSummary(BaseSchema):
    """Summary of an arm (trunk or branch_N) for estimation."""

    arm_idx: int
    name: str
    trajectory_count: int


@dataclass
class EstimationSummary(BaseSchema):
    """Summary of trajectories and arms for easy lookup."""

    trajectories: list[TrajectoryGrouping]
    arms: list[ArmSummary]
