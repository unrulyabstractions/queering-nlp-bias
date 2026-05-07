"""Max-probability core estimator.

The core is the system compliance vector of the trajectory with the
highest sequence-level conditional log probability under this arm's
conditioning (i.e. argmax of `log p(x | arm_prefix)`).

Weights default to uniform so spread metrics (deviance, orientation)
measure how the population deviates from the chosen anchor trajectory.
"""

from __future__ import annotations

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
class MaxProbCoreParams(WeightingMethodParams):
    """Parameters for max-probability core estimation."""

    name: ClassVar[str] = "max-prob"
    description: ClassVar[str] = "max-log-prob-trajectory"


def _uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: MaxProbCoreParams,
) -> list[float]:
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


def _max_prob_core(
    trajs: Sequence[TrajectoryScoringData],
    params: MaxProbCoreParams,
) -> list[float]:
    """Return compliance scores of the trajectory with max conditional log prob."""
    if not trajs:
        raise MethodNotApplicableError("no trajectories")

    arm = trajs[0].arm

    def _lp(t: TrajectoryScoringData) -> float:
        return t.conditional_logprobs.get(arm, float("-inf"))

    best = max(trajs, key=_lp)
    return list(best.structure_scores)


if ENABLED:
    _uniform_weights = register_method(MaxProbCoreParams, core_fn=_max_prob_core)(
        _uniform_weights
    )
