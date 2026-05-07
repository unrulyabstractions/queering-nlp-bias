"""Logit-KDE mode core estimator.

For each structure independently, the core component is the mode of the
distribution of compliance scores across this arm's trajectories,
estimated via a logit-transformed Gaussian KDE.

The logit transform corrects the boundary bias of naive KDE on data
supported in [0, 1]. See `src.common.math.logit_kde` for details.

Weights default to uniform: spread metrics (deviance, orientation)
measure how the population disperses around the per-structure modes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from src.common.math.logit_kde import logit_kde_mode

from ..estimation_structure import TrajectoryScoringData
from ..weighting_method_registry import (
    MethodNotApplicableError,
    WeightingMethodParams,
    register_method,
)

# Set to False to disable this method
ENABLED = True


@dataclass
class ModeCoreParams(WeightingMethodParams):
    """Parameters for logit-KDE mode core estimation."""

    name: ClassVar[str] = "mode"
    description: ClassVar[str] = "logit-kde-mode"


def _uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: ModeCoreParams,
) -> list[float]:
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


def _mode_core(
    trajs: Sequence[TrajectoryScoringData],
    params: ModeCoreParams,
) -> list[float]:
    """Per-structure mode of compliance via logit-KDE."""
    if not trajs:
        raise MethodNotApplicableError("no trajectories")

    n_structures = len(trajs[0].structure_scores)
    return [
        logit_kde_mode([t.structure_scores[i] for t in trajs])
        for i in range(n_structures)
    ]


if ENABLED:
    _uniform_weights = register_method(ModeCoreParams, core_fn=_mode_core)(
        _uniform_weights
    )
