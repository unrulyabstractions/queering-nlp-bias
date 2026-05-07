"""Per-structure median core estimator.

For each structure independently, the core component is the median of
the compliance scores across the arm's trajectories. Robust to extreme
outliers in either direction; for binary-judged structures this is
either 0, 1, or 0.5 (when the population splits evenly).

Weights default to uniform: spread metrics (deviance, orientation)
measure how the population disperses around the per-structure medians.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..estimation_structure import TrajectoryScoringData
from ..weighting_method_registry import (
    MethodNotApplicableError,
    WeightingMethodParams,
    register_method,
)

# Set to False to disable this method
ENABLED = True


@dataclass
class MedianCoreParams(WeightingMethodParams):
    """Parameters for per-structure median core estimation."""

    name: ClassVar[str] = "median"
    description: ClassVar[str] = "per-structure-median"


def _uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: MedianCoreParams,
) -> list[float]:
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


def _median_core(
    trajs: Sequence[TrajectoryScoringData],
    params: MedianCoreParams,
) -> list[float]:
    """Per-structure median of compliance across trajectories."""
    if not trajs:
        raise MethodNotApplicableError("no trajectories")

    n_structures = len(trajs[0].structure_scores)
    return [
        float(statistics.median(t.structure_scores[i] for t in trajs))
        for i in range(n_structures)
    ]


if ENABLED:
    _uniform_weights = register_method(MedianCoreParams, core_fn=_median_core)(
        _uniform_weights
    )
