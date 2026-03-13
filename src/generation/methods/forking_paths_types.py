"""Data structures for forking paths generation method.

This module contains data classes used by the forking paths method.
Separated to avoid circular imports between method and logging modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.base_schema import BaseSchema
from src.inference.generated_trajectory import GeneratedTrajectory


@dataclass
class TopKCandidate(BaseSchema):
    """A candidate token at a position with probability info."""

    token_id: int
    prob: float
    logprob: float


@dataclass
class PositionAnalysis(BaseSchema):
    """Analysis of a single position in the greedy path.

    Contains entropy at the position and top-K candidate tokens.
    """

    position: int
    entropy: float
    greedy_token_id: int
    candidates: list[TopKCandidate]


@dataclass
class QualifyingFork(BaseSchema):
    """A position/candidate pair that qualifies for forking.

    Represents a specific alternate token at a high-entropy position
    that meets the probability threshold for exploration.
    """

    analysis: PositionAnalysis
    candidate: TopKCandidate


@dataclass
class ForkPoint(BaseSchema):
    """A position where we fork from the greedy path.

    Contains the fork position, the alternate token chosen,
    and all continuations sampled from that fork point.
    """

    position: int
    entropy: float
    greedy_token_id: int
    alternate: TopKCandidate
    continuations: list[GeneratedTrajectory]


