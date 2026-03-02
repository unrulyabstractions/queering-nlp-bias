"""Tree structures for trajectory visualization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TreeNode:
    """Node in a trajectory tree."""

    label: str
    prob: float = 1.0
    children: dict[str, TreeNode] = field(default_factory=dict)
    count: int = 1
    has_true_prob: bool = False
    scores: Optional[list[float]] = None
    is_greedy: bool = False
    traj_probs: list[tuple[str, float]] = field(default_factory=list)
    x: float = 0.0
    y: float = 0.0

    def is_leaf(self) -> bool:
        return not self.children

    def is_branching(self) -> bool:
        return len(self.children) > 1


def build_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
    mode: str = "word",
) -> Optional[TreeNode]:
    """Build trajectory tree.

    Args:
        trajectories: List of trajectory dicts with text, probability, per_token_logprobs
        scores: Map from text -> structure scores
        prompt: Root prompt text
        mode: "token", "word", or "phrase"
    """
    if mode == "token":
        if not any(t.get("per_token_logprobs") for t in trajectories):
            return None
        return _build_token_tree(trajectories, scores, prompt)
    elif mode == "phrase":
        return _collapse_chains(_build_word_tree(trajectories, scores, prompt))
    else:
        return _build_word_tree(trajectories, scores, prompt)


def _build_token_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
) -> TreeNode:
    """Build tree from BPE tokens."""
    root = TreeNode(label=prompt, has_true_prob=True)

    for traj in trajectories:
        tokens = traj.get("per_token_logprobs", [])
        if not tokens:
            continue

        text, prob = traj["text"], traj["probability"]
        root.traj_probs.append((text, prob))
        current = root

        for i, tok in enumerate(tokens):
            token, logprob = tok["token"], tok.get("logprob", 0)
            cond_prob = np.exp(logprob) if logprob else prob ** (1 / len(tokens))

            if token not in current.children:
                current.children[token] = TreeNode(
                    label=token, prob=cond_prob, count=0, has_true_prob=True
                )

            child = current.children[token]
            child.count += 1
            child.prob = cond_prob
            child.traj_probs.append((text, prob))

            if i == len(tokens) - 1:
                child.scores = scores.get(text)
                child.is_greedy = traj.get("is_greedy", False)

            current = child

    return root


def _build_word_tree(
    trajectories: list[dict],
    scores: dict[str, list[float]],
    prompt: str,
) -> TreeNode:
    """Build tree from whitespace-split words."""
    root = TreeNode(label=prompt, has_true_prob=True)

    for traj in trajectories:
        text, prob = traj["text"], traj["probability"]
        tokens = traj.get("per_token_logprobs", [])
        word_probs = _compute_word_probs(tokens, text)
        has_tokens = bool(tokens)

        root.traj_probs.append((text, prob))
        current = root

        for i, (word, word_prob) in enumerate(word_probs):
            if word not in current.children:
                current.children[word] = TreeNode(
                    label=word, prob=word_prob, count=0, has_true_prob=has_tokens
                )

            child = current.children[word]
            child.count += 1
            child.prob = word_prob
            child.traj_probs.append((text, prob))

            if i == len(word_probs) - 1:
                child.scores = scores.get(text)
                child.is_greedy = traj.get("is_greedy", False)

            current = child

    return root


def _compute_word_probs(tokens: list[dict], text: str) -> list[tuple[str, float]]:
    """Compute P(word|context) by chaining token probabilities."""
    words = re.findall(r"\S+", text)
    if not words:
        return []
    if not tokens:
        return [(w, 1.0 / len(words)) for w in words]

    token_texts = [t["token"] for t in tokens]
    token_lps = [t.get("logprob", 0) for t in tokens]
    result, token_idx, char_pos = [], 0, 0

    for word in words:
        word_start = text.find(word, char_pos)
        if word_start == -1:
            result.append((word, 1.0 / len(words)))
            continue

        log_prob, reconstructed = 0.0, ""

        while token_idx < len(token_texts) and reconstructed != word:
            tok = token_texts[token_idx].lstrip()
            if tok and word[len(reconstructed) :].startswith(tok):
                reconstructed += tok
                log_prob += token_lps[token_idx]
                token_idx += 1
            elif not token_texts[token_idx].strip():
                token_idx += 1
            else:
                break

        result.append((word, np.exp(log_prob) if log_prob else 1.0 / len(words)))
        char_pos = word_start + len(word)

    return result


def _collapse_chains(node: TreeNode) -> TreeNode:
    """Collapse single-child chains into phrase nodes."""
    labels, probs = [node.label], [node.prob]
    current, has_true = node, node.has_true_prob

    while len(current.children) == 1:
        child = next(iter(current.children.values()))
        labels.append(child.label)
        probs.append(child.prob)
        has_true = has_true and child.has_true_prob
        current = child

    collapsed = TreeNode(
        label=" ".join(labels),
        prob=float(np.prod(probs)),
        count=current.count,
        has_true_prob=has_true,
        scores=current.scores,
        is_greedy=current.is_greedy,
        traj_probs=current.traj_probs,
    )

    for child in current.children.values():
        c = _collapse_chains(child)
        collapsed.children[c.label] = c

    return collapsed
