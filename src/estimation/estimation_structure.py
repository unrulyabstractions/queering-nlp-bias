"""Scoring and trajectory estimation types.

This module defines DATA STRUCTURES for trajectory scoring data
and arm-level estimation results. The computation logic is in
estimation_pipeline.py:compute_arm_estimate().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .estimation_core_types import CoreVariant
from .estimation_weighted_types import WeightedEstimate

# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY SCORING DATA
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrajectoryScoringData(BaseSchema):
    """A trajectory with its structure scores for estimation.

    Attributes:
        traj_idx: Index of this trajectory
        branch: Display name of the branch ("trunk", "branch_1", etc.)
        structure_scores: List of structure scores for each structure
        conditional_logprobs: Log prob of trajectory conditioned on each arm
        n_continuation_tokens: Number of tokens in the continuation
    """

    traj_idx: int
    branch: str
    structure_scores: list[float]
    conditional_logprobs: dict[str, float]
    n_continuation_tokens: int = 0


@dataclass
class TrajectoryEstimate(BaseSchema):
    """Estimation results for a single trajectory."""

    traj_idx: int
    orientation: list[float]  # theta_n(x) = Lambda_n(x) - <Lambda_n>
    deviance: float  # d_n(x) = ||theta_n(x)||


# ══════════════════════════════════════════════════════════════════════════════
# ARM ESTIMATE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArmEstimate(BaseSchema):
    """Estimation results for an arm (trunk or branch_N).

    An "arm" is a conditioning point in the generation tree - either
    the trunk (all trajectories) or a specific branch prefix.

    Results are stored per weighting method in the `estimates` dict.
    Each weighting method (e.g., "prob", "inv-ppl") produces a
    WeightedEstimate with core, deviance, and orientation statistics.
    """

    arm_idx: int
    name: str
    trajectories: list[TrajectoryEstimate]

    # Estimates keyed by weighting method name
    # e.g., {"prob": WeightedEstimate(...), "inv-ppl": WeightedEstimate(...)}
    estimates: dict[str, WeightedEstimate] = field(default_factory=dict)

    def get_estimate(self, method: str = "prob") -> WeightedEstimate | None:
        """Get the estimate for a specific weighting method."""
        return self.estimates.get(method)

    def get_core(self, method: str = "prob") -> list[float]:
        """Get the core for a specific weighting method."""
        est = self.estimates.get(method)
        return est.core if est else []

    def get_deviance_avg(self, method: str = "prob") -> float:
        """Get expected deviance for a specific weighting method."""
        est = self.estimates.get(method)
        return est.deviance_avg if est else 0.0

    def get_deviance_var(self, method: str = "prob") -> float:
        """Get deviance variance for a specific weighting method."""
        est = self.estimates.get(method)
        return est.deviance_var if est else 0.0

    def get_orientation_avg(self, method: str = "prob") -> list[float]:
        """Get expected orientation for a specific weighting method."""
        est = self.estimates.get(method)
        return est.orientation_avg if est else []

    def get_orientation_norm(self, method: str = "prob") -> float:
        """Get orientation norm for a specific weighting method."""
        est = self.estimates.get(method)
        return est.orientation_norm if est else 0.0

    def get_core_variants(self, method: str = "prob") -> list[CoreVariant]:
        """Get core variants for a specific weighting method."""
        est = self.estimates.get(method)
        return est.core_variants if est else []

    def get_core_by_name(self, name: str, method: str = "prob") -> CoreVariant | None:
        """Get a specific core variant by name.

        Args:
            name: Name of the core variant (e.g., "standard", "mode", "confident").
            method: Weighting method (default: "prob").

        Returns:
            CoreVariant if found, None otherwise.
        """
        est = self.estimates.get(method)
        if est:
            return est.get_core_by_name(name)
        return None

    def get_primary_core(self, method: str = "prob") -> list[float]:
        """Get the primary core (standard variant).

        Falls back to the standard core field if the variant is not found.
        """
        variant = self.get_core_by_name("standard", method=method)
        if variant:
            return variant.core
        return self.get_core(method)

    def get_excess_deviance_avg(self, method: str = "prob") -> float:
        """Get expected excess deviance E[∂⁺] for a specific weighting method."""
        est = self.estimates.get(method)
        return est.excess_deviance_avg if est else 0.0

    def get_deficit_deviance_avg(self, method: str = "prob") -> float:
        """Get expected deficit deviance E[∂⁻] for a specific weighting method."""
        est = self.estimates.get(method)
        return est.deficit_deviance_avg if est else 0.0

    def get_mutual_deviance_avg(self, method: str = "prob") -> float:
        """Get expected mutual deviance E[∂_M] for a specific weighting method."""
        est = self.estimates.get(method)
        return est.mutual_deviance_avg if est else 0.0

    def get_core_diversity(self, method: str = "prob") -> float:
        """Get core diversity (Hill D_1) for a specific weighting method."""
        est = self.estimates.get(method)
        return est.core_diversity if est else 0.0
