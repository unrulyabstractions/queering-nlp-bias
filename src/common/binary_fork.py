"""BinaryFork: a pairwise comparison between two branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base_schema import BaseSchema


@dataclass
class BinaryFork(BaseSchema):
    """A pairwise comparison between two branches at a divergence point.

    Attributes:
        next_token_ids: The two token IDs being compared (branch_a, branch_b)
        next_token_logprobs: Log-probabilities for each token
        arm_index: Which arms the two branches belong to (arm_a, arm_b)
    """

    next_token_ids: tuple[int, int]
    next_token_logprobs: tuple[float, float]
    fork_idx: int | None = None  # Index in parent tree's forks tuple
    arm_index: tuple[int, int] | None = None
    analysis: Any | None = None
