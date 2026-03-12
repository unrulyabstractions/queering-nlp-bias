"""Probability weighting method for estimation.

Weights trajectories by their normalized probability p(x).
This is the standard/default weighting scheme.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from src.common.math.probability_utils import normalize_log_probs

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class ProbWeightingParams(WeightingMethodParams):
    """Parameters for probability weighting.

    No configurable parameters - uses raw normalized probabilities.
    """

    name: ClassVar[str] = "prob"
    description: ClassVar[str] = "probability-weighted"


def compute_prob_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: ProbWeightingParams,
) -> list[float]:
    """Compute probability weights from log probabilities.

    Simply normalizes log probabilities to proper probabilities.
    The n_tokens parameter is ignored for this weighting scheme.

    Args:
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory (unused)
        params: Method parameters (unused)

    Returns:
        Normalized probabilities summing to 1.0
    """
    return normalize_log_probs(log_probs)


if ENABLED:
    compute_prob_weights = register_method(ProbWeightingParams)(compute_prob_weights)
