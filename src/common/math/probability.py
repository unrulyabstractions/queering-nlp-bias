"""Probability utilities for log-probability manipulation and normalization.

This module provides unified APIs for working with log probabilities:
- Normalization to proper probability distributions
- Inverse perplexity weighting schemes
- Extraction of conditional log probabilities from structured data

Key functions:
- normalize_log_probs: Convert log probs to normalized probabilities
- normalize_indexed_log_probs: Normalize (index, log_prob) pairs with sorting
- compute_inv_perplexity_weights: Compute 1/perplexity weights for model confidence
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol, TypeVar

# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLS
# ══════════════════════════════════════════════════════════════════════════════


class HasConditionalLogProbs(Protocol):
    """Protocol for items with conditional log probabilities."""

    @property
    def conditional_logprobs(self) -> dict[str, float]: ...


T = TypeVar("T", bound=HasConditionalLogProbs)


# ══════════════════════════════════════════════════════════════════════════════
# LOG PROBABILITY NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════


def normalize_log_probs(log_probs: Sequence[float]) -> list[float]:
    """Convert log probabilities to normalized probabilities.

    Uses the logsumexp trick for numerical stability:
    p_i = exp(log_p_i - max(log_p)) / sum(exp(log_p_j - max(log_p)))

    Args:
        log_probs: Sequence of log probabilities

    Returns:
        List of normalized probabilities summing to 1.0.
        Returns uniform distribution if input is empty or all -inf.

    Example:
        >>> normalize_log_probs([-1.0, -2.0, -3.0])
        [0.665..., 0.245..., 0.090...]  # sums to 1.0
    """
    if not log_probs:
        return []

    # Numerical stability: subtract max before exp
    max_lp = max(log_probs)
    probs = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(probs)

    if total > 0:
        return [p / total for p in probs]

    # Fallback to uniform if all probabilities are 0 (e.g., all -inf)
    return [1.0 / len(probs)] * len(probs)


def normalize_indexed_log_probs(
    indexed_log_probs: Sequence[tuple[int, float]],
    descending: bool = True,
) -> list[tuple[int, float]]:
    """Normalize indexed log probabilities and optionally sort.

    Useful when working with (trajectory_idx, log_prob) pairs.

    Args:
        indexed_log_probs: Sequence of (index, log_probability) tuples
        descending: If True, sort by probability descending (highest first)

    Returns:
        List of (index, normalized_prob) tuples.
        Returns empty list if input is empty.
        Returns uniform distribution if all probabilities are 0.

    Example:
        >>> normalize_indexed_log_probs([(0, -1.0), (1, -2.0)])
        [(0, 0.731...), (1, 0.268...)]  # sorted by prob desc
    """
    if not indexed_log_probs:
        return []

    # Extract and normalize probabilities
    indices = [idx for idx, _ in indexed_log_probs]
    log_probs = [lp for _, lp in indexed_log_probs]
    probs = normalize_log_probs(log_probs)

    # Combine back with indices
    result = list(zip(indices, probs))

    if descending:
        result = sorted(result, key=lambda x: -x[1])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# INVERSE PERPLEXITY WEIGHTING
# ══════════════════════════════════════════════════════════════════════════════


def compute_inv_perplexity_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
) -> list[float]:
    """Compute normalized inverse perplexity weights.

    Inverse perplexity weights sequences by model confidence per token
    rather than raw probability. This downweights sequences that are
    long but low-confidence and upweights short confident sequences.

    inv_ppl = exp(log_prob / n_tokens) = 1/perplexity

    Args:
        log_probs: Sequence of log probabilities
        n_tokens: Number of tokens for each sequence (for normalizing)

    Returns:
        List of normalized inverse perplexity weights summing to 1.0.
        Uses uniform distribution if all weights are 0.

    Raises:
        ValueError: If log_probs and n_tokens have different lengths.

    Example:
        >>> compute_inv_perplexity_weights([-10.0, -20.0], [10, 20])
        [0.5, 0.5]  # both have same per-token probability
    """
    if len(log_probs) != len(n_tokens):
        raise ValueError(
            f"log_probs ({len(log_probs)}) and n_tokens ({len(n_tokens)}) "
            "must have the same length"
        )

    if not log_probs:
        return []

    inv_ppls = []
    for lp, n in zip(log_probs, n_tokens):
        if n > 0 and lp > -700:  # Avoid underflow
            inv_ppls.append(math.exp(lp / n))
        else:
            inv_ppls.append(0.0)

    total = sum(inv_ppls)
    if total > 0:
        return [p / total for p in inv_ppls]

    # Fallback to uniform
    return [1.0 / len(inv_ppls)] * len(inv_ppls)


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL LOG PROBABILITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════


def get_conditional_log_probs(
    items: Sequence[T],
    condition_name: str,
    exclude_branch_mismatch: bool = True,
) -> list[tuple[int, float]]:
    """Extract conditional log probabilities from items with indexed trajectories.

    For items with a `conditional_logprobs` dict and `traj_idx`/`branch` attributes,
    extracts log probs for a specific condition (e.g., "trunk" or branch name).

    Args:
        items: Sequence of items with conditional_logprobs, traj_idx, and branch attrs
        condition_name: Name of the condition to extract (e.g., "trunk", "branch_1")
        exclude_branch_mismatch: If True and condition is not "trunk", skip items
            where branch != condition_name and logprob == 0.0 (not in this branch)

    Returns:
        List of (traj_idx, log_probability) tuples for valid trajectories.

    Note:
        Items must have:
        - conditional_logprobs: dict[str, float]
        - traj_idx: int
        - branch: str (if exclude_branch_mismatch is True)
    """
    result = []
    for item in items:
        lp = item.conditional_logprobs.get(condition_name, 0.0)

        # Skip trajectories not in this arm (logprob = 0.0 marker)
        if exclude_branch_mismatch and condition_name != "trunk":
            branch = getattr(item, "branch", None)
            if lp == 0.0 and branch != condition_name:
                continue

        traj_idx = getattr(item, "traj_idx", None)
        if traj_idx is not None:
            result.append((traj_idx, lp))

    return result
