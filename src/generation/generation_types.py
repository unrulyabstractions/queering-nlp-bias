"""Type definitions for trajectory generation.

This module defines data types for arms and generation results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.inference.generated_trajectory import GeneratedTrajectory


@dataclass
class GenerationArm:
    """An arm configuration for trajectory generation (trunk or branch)."""

    prefill: str  # Full prefill text (skip_prefix + trunk + branch)
    name: str  # Name of this arm
    arm_index: int  # 0 for trunk, N for branch_N


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
    arm_indices: list[int]  # arm_index for each trajectory
    trunk_length: int
    prompt_length: int  # Length of just the prompt (no trunk) in tokens
