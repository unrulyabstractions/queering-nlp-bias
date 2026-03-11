"""Inverse perplexity weighting method for estimation.

Weights trajectories by inverse perplexity: exp(log_p / n_tokens).
This weights by model confidence per token rather than raw probability,
which downweights long low-confidence sequences and upweights short
confident sequences.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from src.common.math.probability import compute_inv_perplexity_weights

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class InvPplWeightingParams(WeightingMethodParams):
    """Parameters for inverse perplexity weighting.

    No configurable parameters - uses standard inv-ppl formula.
    """

    name: ClassVar[str] = "inv-ppl"
    description: ClassVar[str] = "inv-perplexity-weighted"


def compute_inv_ppl_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: InvPplWeightingParams,
) -> list[float]:
    """Compute inverse perplexity weights.

    inv_ppl = exp(log_prob / n_tokens) = 1/perplexity

    This normalizes by sequence length, so:
    - Long sequences with low per-token probability get lower weight
    - Short sequences with high per-token probability get higher weight

    Args:
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory
        params: Method parameters (unused)

    Returns:
        Normalized inverse perplexity weights summing to 1.0
    """
    return compute_inv_perplexity_weights(log_probs, n_tokens)


if ENABLED:
    compute_inv_ppl_weights = register_method(InvPplWeightingParams)(compute_inv_ppl_weights)
