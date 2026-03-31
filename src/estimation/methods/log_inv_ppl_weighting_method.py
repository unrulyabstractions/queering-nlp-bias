"""Log inverse perplexity weighting method for estimation.

Weights trajectories by log(inv-ppl_i) / sum(log(inv-ppl_j)).
where log(inv-ppl) = log_prob / n_tokens.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..weighting_method_registry import WeightingMethodParams, register_method

# Set to False to disable this weighting method
ENABLED = True


@dataclass
class LogInvPplWeightingParams(WeightingMethodParams):
    """Parameters for log inverse perplexity weighting.

    No configurable parameters - uses log(inv-ppl) = log_prob / n_tokens.
    """

    name: ClassVar[str] = "log-inv-ppl"
    description: ClassVar[str] = "log-inv-perplexity-weighted"


def compute_log_inv_ppl_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: LogInvPplWeightingParams,
) -> list[float]:
    """Compute log inverse perplexity weights.

    w_i = log(inv_ppl_i) / sum_j(log(inv_ppl_j))
        = (log_prob_i / n_tokens_i) / sum_j(log_prob_j / n_tokens_j)

    Since log(inv_ppl) values are negative, dividing by the sum (also negative)
    gives positive weights that sum to 1.0.

    Args:
        log_probs: Log probabilities for each trajectory
        n_tokens: Number of tokens per trajectory
        params: Method parameters (unused)

    Returns:
        Normalized weights summing to 1.0
    """
    if len(log_probs) != len(n_tokens):
        raise ValueError(
            f"log_probs ({len(log_probs)}) and n_tokens ({len(n_tokens)}) "
            "must have the same length"
        )

    if not log_probs:
        return []

    # Compute log(inv_ppl) = log_prob / n_tokens for each trajectory
    log_inv_ppls = []
    for lp, n in zip(log_probs, n_tokens):
        if n > 0:
            log_inv_ppls.append(lp / n)
        else:
            # Degenerate case: use raw log_prob
            log_inv_ppls.append(lp)

    total = sum(log_inv_ppls)

    # If total is 0 (all values are 0), use uniform
    if total == 0:
        return [1.0 / len(log_inv_ppls)] * len(log_inv_ppls)

    return [v / total for v in log_inv_ppls]


if ENABLED:
    compute_log_inv_ppl_weights = register_method(LogInvPplWeightingParams)(
        compute_log_inv_ppl_weights
    )
