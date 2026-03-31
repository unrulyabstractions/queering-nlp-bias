"""Log probability weighting method for estimation.

Weights trajectories by log(p_i) / sum(log(p_j)).
Since log probabilities are negative, this gives positive weights
that sum to 1.0.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class LogProbWeightingParams(WeightingMethodParams):
    """Parameters for log probability weighting.

    No configurable parameters - uses raw log probabilities.
    """

    name: ClassVar[str] = "log-prob"
    description: ClassVar[str] = "log-probability-weighted"


def compute_log_prob_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: LogProbWeightingParams,
) -> list[float]:
    """Compute log probability weights.

    w_i = log(p_i) / sum_j(log(p_j))

    Since log probabilities are negative, dividing by the sum (also negative)
    gives positive weights. The n_tokens parameter is ignored.

    Args:
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory (unused)
        params: Method parameters (unused)

    Returns:
        Normalized weights summing to 1.0
    """
    if not log_probs:
        return []

    total = sum(log_probs)

    # If total is 0 (all log_probs are 0), use uniform
    if total == 0:
        return [1.0 / len(log_probs)] * len(log_probs)

    return [lp / total for lp in log_probs]


if ENABLED:
    compute_log_prob_weights = register_method(LogProbWeightingParams)(
        compute_log_prob_weights
    )
