"""Common experiment types used across generation, scoring, and estimation.

This module defines shared data types for the experiment pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.inference.generated_trajectory import GeneratedTrajectory


@dataclass
class GenerationArm:
    """An arm configuration for trajectory generation.

    Arms are stored in a list - position IS the index.
    parent_idx points to parent arm's position for AfterBranch text selection.
    """

    name: str
    prefill: str
    parent_idx: int | None = None

    def to_dict(self) -> dict:
        """Convert to dict for JSON storage."""
        return {
            "name": self.name,
            "prefill": self.prefill,
            "parent_idx": self.parent_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GenerationArm":
        """Create from dict."""
        return cls(
            name=data["name"],
            prefill=data["prefill"],
            parent_idx=data.get("parent_idx"),
        )


@dataclass
class OutputPaths:
    """Computed output paths for the full experiment pipeline."""

    generation: Path
    judgment: Path
    estimation: Path


@dataclass
class ArmGenerationResult:
    """Result from generating trajectories across all branches."""

    trajectories: list[GeneratedTrajectory]
    arm_indices: list[int]  # arm_idx for each trajectory
    arm_token_lengths: list[int]  # Total tokens (prompt + prefill) for each arm
    arms: list[GenerationArm]  # Full arm objects with prefills

    @property
    def arm_names(self) -> list[str]:
        """Arm names in index order (derived from arms)."""
        return [arm.name for arm in self.arms]

    @property
    def prompt_length(self) -> int:
        """Length of just the prompt (root arm) in tokens."""
        return self.arm_token_lengths[0] if self.arm_token_lengths else 0

    @property
    def trunk_length(self) -> int:
        """Length of prompt + trunk (trunk arm) in tokens."""
        if self.arm_token_lengths and len(self.arm_token_lengths) > 1:
            return self.arm_token_lengths[1]
        return self.prompt_length
