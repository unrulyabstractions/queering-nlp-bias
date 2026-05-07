"""Max-inverse-perplexity core estimator.

The core is the system compliance vector of the trajectory with the
highest per-token confidence (argmax of `log p(x | arm_prefix) / n_tokens`).

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
class MaxInvPplCoreParams(WeightingMethodParams):
    """Parameters for max-inverse-perplexity core estimation."""

    name: ClassVar[str] = "max-inv-ppl"
    description: ClassVar[str] = "max-inv-perplexity-trajectory"


def _uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: MaxInvPplCoreParams,
) -> list[float]:
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


def _max_inv_ppl_core(
    trajs: Sequence[TrajectoryScoringData],
    params: MaxInvPplCoreParams,
) -> list[float]:
    """Return compliance scores of the trajectory with max inv-perplexity."""
    if not trajs:
        raise MethodNotApplicableError("no trajectories")

    arm = trajs[0].arm

    def _per_token_lp(t: TrajectoryScoringData) -> float:
        n = t.n_generated_tokens
        lp = t.conditional_logprobs.get(arm, float("-inf"))
        if n <= 0:
            return float("-inf")
        return lp / n

    best = max(trajs, key=_per_token_lp)
    return list(best.structure_scores)


if ENABLED:
    _uniform_weights = register_method(
        MaxInvPplCoreParams, core_fn=_max_inv_ppl_core
    )(_uniform_weights)
