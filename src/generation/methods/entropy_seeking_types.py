"""Data structures for entropy-seeking generation method.

This module contains data classes used by the entropy-seeking method.
Separated to avoid circular imports between method and logging modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.inference.generated_trajectory import GeneratedTrajectory


@dataclass
class BestPosition:
    """Result of finding the best unused position in a path."""

    position: int | None
    entropy: float


@dataclass
class TreePath:
    """A path in the entropy-seeking tree with precomputed entropies.

    Represents a single trajectory with its entropy values at each position,
    tracking which positions have been used for expansion.
    """

    trajectory: GeneratedTrajectory
    path_id: int
    entropies: list[float]
    continuation: str = ""  # For display
    parent_id: int | None = None  # Which path we branched from
    branch_pos: int | None = None  # Position where we branched
    used_positions: set[int] = field(default_factory=set)

    @property
    def token_ids(self) -> list[int]:
        return self.trajectory.token_ids

    @property
    def max_entropy(self) -> float:
        return max(self.entropies) if self.entropies else 0.0

    def prefix(self, position: int) -> list[int]:
        """Get tokens up to and including position."""
        return self.token_ids[: position + 1]

    def mark_used(self, pos: int) -> None:
        """Mark a position as used for splitting."""
        self.used_positions.add(pos)

    def best_unused_position(self, prompt_len: int) -> BestPosition:
        """Find highest-entropy unused position."""
        best_pos = None
        best_entropy = -math.inf

        for i, entropy in enumerate(self.entropies):
            pos = prompt_len + i
            if pos not in self.used_positions and entropy > best_entropy:
                best_entropy = entropy
                best_pos = pos

        return BestPosition(position=best_pos, entropy=best_entropy)


@dataclass
class ExpansionPoint:
    """The best expansion point across all paths."""

    path: TreePath | None
    position: int | None
    entropy: float
