"""Greedy decoding core estimator.

The core is the system compliance vector of the trajectory that the
sample collectively votes "most greedy" via a token-by-token majority
walk through the empirical prefix tree.

Algorithm
---------
Within an arm, all trajectories share the same prefill, so we walk only
the generated continuations:

1. Start with the full set of trajectories as candidates.
2. At position 0, look at each candidate's first continuation token.
   The locally-greedy choice is the *most-frequent* token at that
   position. Keep only the candidates that emitted it.
3. Advance one position and repeat: among the remaining candidates,
   pick the most frequent next token; keep its owners.
4. Stop when a single candidate remains, or when all remaining
   candidates have run out of tokens (they share the entire greedy
   path — pick any).

This is genuinely distinct from `max-prob` and `max-inv-ppl`: a sample
might have the highest per-token confidence yet diverge from the
empirical greedy path early. By walking node-by-node, we recover the
trajectory the samples *jointly* point to, even if it isn't a single
sample's likelihood maximum.

Selection precedence
--------------------
* If any trajectory carries `is_greedy=True` (explicit greedy
  generation), use it.
* Else, if continuation token streams are available, walk the prefix
  tree as above.
* Otherwise raise `MethodNotApplicableError` — there's no honest way to
  pick "the greedy" trajectory from sequence-level summaries alone.

Weights default to uniform so spread metrics (deviance, orientation)
remain informative against the population.
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
class GreedyCoreParams(WeightingMethodParams):
    """Parameters for greedy core estimation. No configurable params."""

    name: ClassVar[str] = "greedy"
    description: ClassVar[str] = "greedy-decoding-trajectory"


def _uniform_weights(
    log_probs: Sequence[float],
    n_tokens: Sequence[int],
    params: GreedyCoreParams,
) -> list[float]:
    """Uniform weights — used only for spread metrics (deviance, orientation)."""
    n = len(log_probs)
    if n == 0:
        return []
    return [1.0 / n] * n


def _walk_greedy(token_streams: list[list[int]]) -> int:
    """Token-by-token majority walk. Returns the chosen trajectory index.

    Tied next-tokens are broken by smallest token id for determinism.
    Among trajectories that share the entire greedy path (rare), the
    longest is returned (it carried the walk further).
    """
    n = len(token_streams)
    if n == 1:
        return 0

    candidates = list(range(n))
    pos = 0

    while len(candidates) > 1:
        counts: dict[int, int] = {}
        owners: dict[int, list[int]] = {}
        for c in candidates:
            seq = token_streams[c]
            if pos >= len(seq):
                continue
            tok = seq[pos]
            counts[tok] = counts.get(tok, 0) + 1
            owners.setdefault(tok, []).append(c)

        if not counts:
            # All remaining candidates have ended at this position — they
            # share the greedy path entirely.
            break

        best_count = max(counts.values())
        best_tokens = [tok for tok, c in counts.items() if c == best_count]
        best_token = min(best_tokens)
        candidates = owners[best_token]
        pos += 1

    return max(candidates, key=lambda c: len(token_streams[c]))


def _greedy_core(
    trajs: Sequence[TrajectoryScoringData],
    params: GreedyCoreParams,
) -> list[float]:
    """Return compliance scores of the greedy (or empirically-greedy) trajectory."""
    if not trajs:
        raise MethodNotApplicableError("no trajectories")

    for t in trajs:
        if t.is_greedy:
            return list(t.structure_scores)

    streams = [list(t.continuation_token_ids) for t in trajs]
    if not any(streams):
        raise MethodNotApplicableError(
            "no token-level continuations available; greedy walk needs per-trajectory "
            "token_ids (typically loaded from generation.json)"
        )

    chosen = _walk_greedy(streams)
    return list(trajs[chosen].structure_scores)


if ENABLED:
    _uniform_weights = register_method(GreedyCoreParams, core_fn=_greedy_core)(
        _uniform_weights
    )
