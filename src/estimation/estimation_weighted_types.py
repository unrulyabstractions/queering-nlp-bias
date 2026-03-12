"""Weighted estimation result types.

This module defines data structures for storing estimation results
from a single weighting method. ArmEstimate aggregates these across
all registered weighting methods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .estimation_core_types import CoreVariant


@dataclass
class WeightedEstimate(BaseSchema):
    """Estimation results for ONE weighting method.

    Contains all metrics computed using a specific weighting scheme:
    - Core: weighted center of compliance vectors
    - Deviance: spread around core (both this arm's and trunk's)
    - Orientation: direction from reference (trunk) core
    - Core variants: different (q, r) parameterizations
    """

    # The weighting method used
    method_name: str

    # Primary core (q=1, r=1)
    core: list[float]

    # Deviance stats relative to THIS arm's core: E[d|branch]
    deviance_avg: float  # E[d|B]
    deviance_var: float  # Var[d|B]

    # Deviance stats relative to TRUNK core: E[d|trunk]
    deviance_avg_trunk: float = 0.0  # E[d|T]

    # Expected deviance delta: E[Δd] = E[d|branch] - E[d|trunk]
    deviance_delta: float = 0.0  # E[Δd]

    # Expected orientation relative to TRUNK core: E[θ|trunk]
    orientation_avg: list[float] = field(default_factory=list)

    # Distance between this arm's core and trunk core: ||E[θ|trunk]||
    orientation_norm: float = 0.0

    # Excess deviance: E[∂⁺] - how much samples over-comply
    excess_deviance_avg: float = 0.0

    # Deficit deviance: E[∂⁻] - how much samples under-comply
    deficit_deviance_avg: float = 0.0

    # Mutual deviance: E[∂_M] - symmetric deviance using JS divergence
    mutual_deviance_avg: float = 0.0

    # Core diversity: effective number of structures (Hill number D_1)
    core_diversity: float = 0.0

    # All (q, r) variants for this weighting scheme
    core_variants: list[CoreVariant] = field(default_factory=list)

    def get_core_by_name(self, name: str) -> CoreVariant | None:
        """Get a specific core variant by name."""
        for v in self.core_variants:
            if v.name == name:
                return v
        return None

    @classmethod
    def empty(cls, method_name: str) -> WeightedEstimate:
        """Create an empty estimate for the given method."""
        return cls(
            method_name=method_name,
            core=[],
            deviance_avg=0.0,
            deviance_var=0.0,
        )
