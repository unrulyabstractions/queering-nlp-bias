"""TokenTrajectory: a sequence of tokens with logprobs and logits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

import torch

from .base_schema import BaseSchema
from .text.thinking_filter import strip_thinking_blocks
from .viz_utils import sanitize_floats

T = TypeVar("T", bound="TokenTrajectory")


@dataclass
class TokenTrajectory(BaseSchema):
    """A sequence of tokens with associated logprobs and logits.

    All arrays have length n_sequence (full sequence length).
    The first token has logprob=0 (probability 1, since it's given).
    """

    token_ids: list[int]
    logprobs: list[float]
    logits: list[float]
    full_logits: torch.Tensor | None = None
    entropies: list[float] | None = None  # Per-position entropy (for generated tokens)

    # Text fields (set by generate_trajectory_from_prompt)
    prefill_text: str | None = None  # Trunk/branch/twig text prepended before generation
    generated_text: str | None = None  # Text the model actually generated

    # Per-arm lengths (set by GenerationOutput.from_tree, indices match arm list)
    arm_token_lengths: list[int] | None = None  # Token count for each arm's prefill
    arm_text_lengths: list[int] | None = None  # Char count for each arm's prefill

    traj_idx: int | None = None  # Index in parent tree's trajs tuple
    nodes_idx: tuple[int, ...] | None = None
    arm_idx: tuple[int, ...] | None = None
    prefill_length: int | None = None  # Token position where generated content starts

    @property
    def continuation_text(self) -> str | None:
        """Full continuation = prefill + generated."""
        if self.generated_text is None:
            return None
        prefill = self.prefill_text or ""
        return prefill + self.generated_text

    @property
    def continuation_text_no_thinking(self) -> str | None:
        """Full continuation with <think>...</think> blocks removed."""
        text = self.continuation_text
        if text is None:
            return None
        return strip_thinking_blocks(text)

    def text_after_arm(self, arm_idx: int) -> str:
        """Get continuation text after a specific arm's prefill.

        Uses precomputed arm_text_lengths to slice without parsing.
        """
        continuation = self.continuation_text_no_thinking or ""
        if self.arm_text_lengths is None or arm_idx >= len(self.arm_text_lengths):
            return continuation
        offset = self.arm_text_lengths[arm_idx]
        return continuation[offset:]

    def can_have_internals(self) -> bool:
        return False

    def has_internals(self) -> bool:
        return False

    def has_internals_for(self, names_filter: callable | None = None) -> bool:
        return False

    @property
    def n_sequence(self) -> int:
        return len(self.token_ids)

    @property
    def sequence_length(self) -> int:
        return self.n_sequence

    @property
    def length(self) -> int:
        return self.n_sequence

    @property
    def n_pred(self) -> int:
        return max(0, self.n_sequence - 1)

    @property
    def predictions_length(self) -> int:
        return self.n_pred

    @property
    def pred_token_ids(self) -> list[int]:
        return self.token_ids[1:]

    @property
    def pred_logprobs(self) -> list[float]:
        return self.logprobs[1:]

    @property
    def pred_logits(self) -> list[float]:
        return self.logits[1:]

    @property
    def pred_full_logits(self) -> torch.Tensor | None:
        if self.full_logits is None:
            return None
        return self.full_logits[1:]

    @property
    def next_token_logprob_sequence(self) -> list[float]:
        return self.pred_logprobs

    @property
    def branching_points(self) -> list[int]:
        if self.nodes_idx is None:
            return []
        return list(getattr(self, "_branching_positions", []))

    def sanitize(self: T) -> T:
        """Sanitize float values (replace NaN/inf) for JSON serialization."""
        self.logprobs = sanitize_floats(self.logprobs)
        self.logits = sanitize_floats(self.logits)
        return self

    def pop_heavy(self) -> None:
        self.pop_full_logits()

    def pop_full_logits(self) -> torch.Tensor | None:
        seq = self.full_logits
        self.full_logits = None
        return seq

    def to_dict(self) -> dict:
        full_logits = self.pop_full_logits()
        d = super().to_dict()
        self.full_logits = full_logits
        return d

    def get_conditional_prob(
        self, start_token_ids_pos: int, end_token_ids_pos: int
    ) -> float | None:
        if (
            start_token_ids_pos < 0
            or end_token_ids_pos > self.length
            or start_token_ids_pos >= end_token_ids_pos
        ):
            return None
        log_prob_sum = sum(self.logprobs[start_token_ids_pos:end_token_ids_pos])
        return math.exp(log_prob_sum)
