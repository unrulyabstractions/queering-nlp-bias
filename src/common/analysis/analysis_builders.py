"""Builder functions for analysis objects.

Provides functions to build analysis objects from tree components.
"""

from __future__ import annotations

import math

import torch

from ..binary_fork import BinaryFork
from ..branching_node import BranchingNode
from ..math import (
    log_odds,
    logprob_to_prob,
    probability_ratio,
    q_fork_concentration,
    q_fork_diversity,
    q_fork_entropy,
    vocab_entropy_from_logits,
)
from ..token_tree import TokenTree
from .metrics import (
    ForkAnalysis,
    ForkMetrics,
    NodeAnalysis,
    NodeMetrics,
)


def build_fork_analysis(fork_idx: int, fork: BinaryFork) -> ForkAnalysis:
    """Build analysis for a binary fork.

    Args:
        fork_idx: Index of the fork in the tree
        fork: The BinaryFork to analyze

    Returns:
        ForkAnalysis containing fork metrics
    """
    lp_a, lp_b = fork.next_token_logprobs
    p_a, p_b = logprob_to_prob(lp_a), logprob_to_prob(lp_b)

    return ForkAnalysis(
        fork_idx=fork_idx,
        metrics=ForkMetrics(
            next_token_logprobs=(lp_a, lp_b),
            fork_entropy=q_fork_entropy(p_a, p_b, q=1.0),
            fork_diversity=q_fork_diversity(p_a, p_b, q=1.0),
            fork_simpson=q_fork_diversity(p_a, p_b, q=2.0),
            fork_concentration=q_fork_concentration(p_a, p_b, q=1.0),
            probability_ratio=probability_ratio(p_a, p_b),
            log_odds=log_odds(p_a, p_b),
            logit_diff=lp_a - lp_b,
            reciprocal_rank_a=1.0 if lp_a >= lp_b else 0.5,
        ),
    )


def build_node_analysis(
    node_idx: int,
    node: BranchingNode,
    tree: TokenTree,
) -> NodeAnalysis:
    """Build analysis for a branching node.

    Args:
        node_idx: Index of the node in the tree
        node: The BranchingNode to analyze
        tree: The parent TokenTree (for logits lookup)

    Returns:
        NodeAnalysis containing node metrics
    """
    next_token_logprobs = [float(lp) for lp in node.next_token_logprobs]

    # Use vocab_logits stored on node if available, fallback to tree lookup
    if node.vocab_logits is not None:
        logits = torch.tensor(node.vocab_logits)
        v_entropy = vocab_entropy_from_logits(logits).item()
    else:
        pos = node.branching_token_position
        logits = tree.get_logits_at_node(node_idx, pos)
        v_entropy = (
            vocab_entropy_from_logits(logits).item() if logits is not None else 0.0
        )

    return NodeAnalysis(
        node_idx=node_idx,
        metrics=NodeMetrics(
            next_token_logprobs=next_token_logprobs,
            vocab_entropy=v_entropy,
            vocab_diversity=math.exp(v_entropy),
        ),
    )
