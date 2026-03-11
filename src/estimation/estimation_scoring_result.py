"""Scoring result types for structure-based scoring.

This module defines data structures for organizing and querying
scoring results across different structure types (categorical,
graded, similarity).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema


@dataclass
class ScoreComputation(BaseSchema):
    """Result of computing scores for a structure."""

    aggregate: float
    item_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BundledScoreResult(BaseSchema):
    """Score result for a bundled structure (multiple items aggregated)."""

    aggregate: float  # Overall aggregated score
    items: dict[str, float] = field(default_factory=dict)  # item_name -> score


@dataclass
class StructureScoresResult(BaseSchema):
    """Result of scoring all structures for a set of trajectories."""

    simple_scoring: dict[str, float] = field(default_factory=dict)  # label -> score
    bundled_scoring: dict[str, BundledScoreResult] = field(default_factory=dict)

    def get_score(self, label: str) -> float:
        """Get aggregate score for a structure (simple or bundled)."""
        if label in self.simple_scoring:
            return self.simple_scoring[label]
        if label in self.bundled_scoring:
            return self.bundled_scoring[label].aggregate
        return 0.0

    def get_item_scores(self, label: str) -> dict[str, float]:
        """Get per-item scores for a bundled structure."""
        if label in self.bundled_scoring:
            return self.bundled_scoring[label].items
        return {}

    def all_scores(self) -> dict[str, float]:
        """Get all aggregate scores (simple + bundled) as a flat dict."""
        result = dict(self.simple_scoring)
        for label, bundled in self.bundled_scoring.items():
            result[label] = bundled.aggregate
        return result

    def all_item_scores(self) -> dict[str, dict[str, float]]:
        """Get all item scores for bundled structures."""
        return {
            label: bundled.items for label, bundled in self.bundled_scoring.items()
        }


@dataclass
class StructureInfo(BaseSchema):
    """Information about a scoring structure."""

    idx: int  # Structure index (0-based)
    label: str  # Short label like "c1", "c2", "s1"
    description: str  # Full question or reference text
    is_bundled: bool  # Whether this is a bundled structure (multiple questions)
    question_count: int  # Number of questions (1 for single, N for bundled)
    questions: list[str]  # Individual questions (for bundled structures)
    method_name: str = ""  # Name of the scoring method (e.g., "categorical", "graded")
    struct_idx_in_method: int = 0  # Index within the method's items list


@dataclass
class ArmScoring(BaseSchema):
    """Per-structure compliance scores for an arm."""

    branch: str
    branch_idx: int
    trajectory_count: int
    simple_scoring: dict[str, float] = field(default_factory=dict)  # label -> score
    bundled_scoring: dict[str, BundledScoreResult] = field(default_factory=dict)

    @classmethod
    def from_scores_result(
        cls,
        branch: str,
        branch_idx: int,
        trajectory_count: int,
        scores: StructureScoresResult,
    ) -> ArmScoring:
        """Create from a StructureScoresResult."""
        return cls(
            branch=branch,
            branch_idx=branch_idx,
            trajectory_count=trajectory_count,
            simple_scoring=scores.simple_scoring,
            bundled_scoring=scores.bundled_scoring,
        )

    @property
    def structure_rates(self) -> dict[str, float]:
        """Get all structure rates as flat dict (backward compat)."""
        return {
            **self.simple_scoring,
            **{k: v.aggregate for k, v in self.bundled_scoring.items()},
        }

    @property
    def question_rates(self) -> dict[str, dict[str, float]]:
        """Get bundled item rates (backward compat)."""
        return {k: v.items for k, v in self.bundled_scoring.items()}
