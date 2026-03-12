"""Weighted estimation result types.

This module defines data structures for storing estimation results
from a single weighting method. ArmEstimate aggregates these across
all registered weighting methods.

Orientation vectors are pre-computed and stored here:
- orientation_from_root = arm_core - root_core
- orientation_from_trunk = arm_core - trunk_core
- orientation_from_parent = arm_core - parent_branch_core (for twigs only)

Deviance deltas can be computed from deviance_avg - deviance_avg_trunk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.common.base_schema import BaseSchema

from .estimation_core_types import CoreVariant


@dataclass
class WeightedEstimate(BaseSchema):
    """Estimation results for ONE weighting method.

    Contains metrics that require weighted computation:
    - Core: weighted center of compliance vectors
    - Deviance: spread around cores (requires weights)
    - Core variants: different (q, r) parameterizations
    """

    # The weighting method used
    method_name: str

    # Primary core (q=1, r=1)
    core: list[float]

    # Deviance stats relative to THIS arm's core
    deviance_avg: float  # E[∂|self]
    deviance_var: float  # Var[∂|self]

    # Deviance relative to reference cores (requires weights to compute)
    deviance_avg_root: float = 0.0  # E[∂|root]
    deviance_avg_trunk: float = 0.0  # E[∂|trunk]

    # Directional deviance metrics
    excess_deviance_avg: float = 0.0  # E[∂⁺] - over-compliance
    deficit_deviance_avg: float = 0.0  # E[∂⁻] - under-compliance
    mutual_deviance_avg: float = 0.0  # E[∂_M] - symmetric (JS divergence)

    # Core diversity: effective number of structures (Hill number D_1)
    core_diversity: float = 0.0

    # Orientation relative to reference cores (pre-computed)
    # Orientation = core - reference_core (vector difference)
    orientation_from_root: list[float] = field(default_factory=list)
    orientation_norm_from_root: float = 0.0

    orientation_from_trunk: list[float] = field(default_factory=list)
    orientation_norm_from_trunk: float = 0.0

    # For twigs: orientation relative to parent branch core
    orientation_from_parent: list[float] = field(default_factory=list)
    orientation_norm_from_parent: float = 0.0

    # All (q, r) variants for this weighting scheme
    core_variants: list[CoreVariant] = field(default_factory=list)

    def get_core_by_name(self, name: str) -> CoreVariant | None:
        """Get a specific core variant by name."""
        for v in self.core_variants:
            if v.name == name:
                return v
        return None
