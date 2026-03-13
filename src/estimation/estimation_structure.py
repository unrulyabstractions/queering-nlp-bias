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
        arm: Name of the arm ("root", "trunk", "branch_1", "twig_1_b1", etc.)
        structure_scores: List of structure scores for each structure
        conditional_logprobs: Log prob of trajectory conditioned on each arm
        n_generated_tokens: Number of tokens in the continuation
        text: The continuation text (used for dynamics analysis)
    """

    traj_idx: int
    arm: str
    structure_scores: list[float]
    conditional_logprobs: dict[str, float]
    n_generated_tokens: int = 0
    text: str = ""


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

    def _get_estimate_or_raise(self, method: str) -> WeightedEstimate:
        """Get estimate for method, raising KeyError if not found."""
        est = self.estimates.get(method)
        if est is None:
            raise KeyError(
                f"Weighting method '{method}' not found in arm '{self.name}'. "
                f"Available methods: {list(self.estimates.keys())}"
            )
        return est

    def get_estimate(self, method: str = "prob") -> WeightedEstimate:
        """Get the estimate for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method)

    def get_core(self, method: str = "prob") -> list[float]:
        """Get the core for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).core

    def get_deviance_avg(self, method: str = "prob") -> float:
        """Get expected deviance for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).deviance_avg

    def get_deviance_var(self, method: str = "prob") -> float:
        """Get deviance variance for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).deviance_var

    def get_core_variants(self, method: str = "prob") -> list[CoreVariant]:
        """Get core variants for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).core_variants

    def get_core_by_name(self, name: str, method: str = "prob") -> CoreVariant | None:
        """Get a specific core variant by name.

        Args:
            name: Name of the core variant (e.g., "standard", "mode", "confident").
            method: Weighting method (default: "prob").

        Returns:
            CoreVariant if found, None otherwise.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).get_core_by_name(name)

    def get_primary_core(self, method: str = "prob") -> list[float]:
        """Get the primary core (standard variant).

        Falls back to the standard core field if the variant is not found.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        variant = self.get_core_by_name("standard", method=method)
        if variant:
            return variant.core
        return self.get_core(method)

    def get_excess_deviance_avg(self, method: str = "prob") -> float:
        """Get expected excess deviance E[∂⁺] for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).excess_deviance_avg

    def get_deficit_deviance_avg(self, method: str = "prob") -> float:
        """Get expected deficit deviance E[∂⁻] for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).deficit_deviance_avg

    def get_mutual_deviance_avg(self, method: str = "prob") -> float:
        """Get expected mutual deviance E[∂_M] for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).mutual_deviance_avg

    def get_core_diversity(self, method: str = "prob") -> float:
        """Get core diversity (Hill D_1) for a specific weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).core_diversity

    def get_orientation_from_root(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to root core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_from_root

    def get_orientation_norm_from_root(self, method: str = "prob") -> float:
        """Get orientation norm (magnitude) relative to root core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_norm_from_root

    def get_orientation_from_trunk(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to trunk core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_from_trunk

    def get_orientation_norm_from_trunk(self, method: str = "prob") -> float:
        """Get orientation norm (magnitude) relative to trunk core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_norm_from_trunk

    def get_orientation_from_parent(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to parent branch core (for twigs).

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_from_parent

    def get_orientation_norm_from_parent(self, method: str = "prob") -> float:
        """Get orientation norm relative to parent branch core (for twigs).

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        return self._get_estimate_or_raise(method).orientation_norm_from_parent
