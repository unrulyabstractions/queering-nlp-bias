"""Experiment result dataclasses for estimation.

These types are used for loading and processing estimation results
from JSON output files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.common.base_schema import BaseSchema
from src.common.schema_utils import safe_float

# ══════════════════════════════════════════════════════════════════════════════
# Result Dataclasses
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EstimationArmResult(BaseSchema):
    """Statistics for a single arm (trunk or branch_N).

    Stores estimates per weighting method in a dict.
    """

    name: str
    n_trajectories: int
    # Estimates keyed by weighting method name
    # e.g., {"prob": {...}, "inv-ppl": {...}}
    estimates: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _get_estimate_or_raise(self, method: str) -> dict[str, Any]:
        """Get estimate dict for method, raising KeyError if not found."""
        if method not in self.estimates:
            raise KeyError(
                f"Weighting method '{method}' not found in arm '{self.name}'. "
                f"Available methods: {list(self.estimates.keys())}"
            )
        return self.estimates[method]

    def get_core(self, method: str = "prob") -> list[float]:
        """Get core for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        core = est.get("core")
        if core is None:
            raise KeyError(f"Missing 'core' in estimate for method '{method}'")
        return core

    def get_deviance_avg(self, method: str = "prob") -> float:
        """Get E[d|B] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("deviance_avg"), 0.0)

    def get_deviance_avg_trunk(self, method: str = "prob") -> float:
        """Get E[∂|trunk] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("deviance_avg_trunk"), 0.0)

    def get_deviance_avg_root(self, method: str = "prob") -> float:
        """Get E[∂|root] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("deviance_avg_root"), 0.0)

    def get_core_variants(self, method: str = "prob") -> list[dict[str, Any]]:
        """Get core variants for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return est.get("core_variants", [])

    def get_excess_deviance_avg(self, method: str = "prob") -> float:
        """Get E[∂⁺] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("excess_deviance_avg"), 0.0)

    def get_deficit_deviance_avg(self, method: str = "prob") -> float:
        """Get E[∂⁻] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("deficit_deviance_avg"), 0.0)

    def get_mutual_deviance_avg(self, method: str = "prob") -> float:
        """Get E[∂_M] for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("mutual_deviance_avg"), 0.0)

    def get_core_diversity(self, method: str = "prob") -> float:
        """Get core diversity (Hill D_1) for a weighting method.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("core_diversity"), 0.0)

    def get_orientation_from_root(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to root core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return est.get("orientation_from_root", [])

    def get_orientation_norm_from_root(self, method: str = "prob") -> float:
        """Get orientation norm (magnitude) relative to root core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("orientation_norm_from_root"), 0.0)

    def get_orientation_from_trunk(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to trunk core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return est.get("orientation_from_trunk", [])

    def get_orientation_norm_from_trunk(self, method: str = "prob") -> float:
        """Get orientation norm (magnitude) relative to trunk core.

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("orientation_norm_from_trunk"), 0.0)

    def get_orientation_from_parent(self, method: str = "prob") -> list[float]:
        """Get orientation vector relative to parent branch core (for twigs).

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return est.get("orientation_from_parent", [])

    def get_orientation_norm_from_parent(self, method: str = "prob") -> float:
        """Get orientation norm relative to parent branch core (for twigs).

        Raises:
            KeyError: If the weighting method doesn't exist.
        """
        est = self._get_estimate_or_raise(method)
        return safe_float(est.get("orientation_norm_from_parent"), 0.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any], index: int = 0) -> EstimationArmResult:
        """Create from estimation output dict."""
        trajectories = data.get("trajectories", [])
        n_traj = len(trajectories)
        estimates = data.get("estimates", {})

        return cls(
            name=data.get("name", f"arm_{index}"),
            n_trajectories=n_traj,
            estimates=estimates,
        )


@dataclass
class EstimationResult(BaseSchema):
    """Result of a single experiment run."""

    method: str
    paths: Any  # OutputPaths - avoiding circular import
    arms: list[EstimationArmResult] = field(default_factory=list)
    arm_scoring: list[dict[str, Any]] = field(default_factory=list)
    structure_info: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_trajectories(self) -> int:
        """Total trajectories (trunk count includes all)."""
        return self.trunk.n_trajectories if self.arms else 0

    @property
    def trunk(self) -> EstimationArmResult:
        """Get trunk arm (index 0)."""
        return self.arms[0]

    @classmethod
    def from_estimation_file(
        cls,
        method: str,
        paths: Any,  # OutputPaths
    ) -> EstimationResult:
        """Load result from estimation output file (v2.0 format)."""
        with open(paths.estimation) as f:
            est_data = json.load(f)

        arms = [EstimationArmResult.from_dict(a, i) for i, a in enumerate(est_data["arms"])]

        return cls(
            method=method,
            paths=paths,
            arms=arms,
            arm_scoring=est_data.get("arm_scoring", []),
            structure_info=est_data["structures"],
        )
