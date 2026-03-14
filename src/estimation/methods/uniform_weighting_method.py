"""Uniform weighting method for estimation.

Assigns equal weight to all trajectories regardless of probability.
This serves as a baseline for comparison with probability-based methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class UniformWeightingParams(WeightingMethodParams):
    """Parameters for uniform weighting.

    No configurable parameters - all trajectories get equal weight.
    """

    name: ClassVar[str] = "uniform"
    description: ClassVar[str] = "uniform-weighted"


def compute_uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: UniformWeightingParams,
) -> list[float]:
    """Compute uniform weights (equal for all trajectories).

    Ignores log_probs and n_tokens entirely - each trajectory
    contributes equally to the core estimate.

    Args:
        log_probs: Log probabilities (ignored)
        n_tokens: Token counts (ignored)
        params: Method parameters (unused)

    Returns:
        Uniform weights [1/n, 1/n, ..., 1/n] summing to 1.0
    """
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


if ENABLED:
    compute_uniform_weights = register_method(UniformWeightingParams)(
        compute_uniform_weights
    )
