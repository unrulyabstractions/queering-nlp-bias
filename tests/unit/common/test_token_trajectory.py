"""Tests for TokenTrajectory class."""

from __future__ import annotations

import math

import pytest
import torch

from src.common.token_trajectory import TokenTrajectory
from fixtures.sample_trajectories import create_sample_trajectory


class TestTokenTrajectoryProperties:
    """Tests for TokenTrajectory computed properties."""

    def test_n_sequence(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.n_sequence == len(sample_token_ids)

    def test_sequence_length_alias(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.sequence_length == traj.n_sequence

    def test_length_alias(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.length == traj.n_sequence

    def test_n_pred(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.n_pred == len(sample_token_ids) - 1

    def test_n_pred_empty_sequence(self):
        traj = TokenTrajectory(
            token_ids=[1],  # Single token
            logprobs=[0.0],
            logits=[0.0],
        )
        assert traj.n_pred == 0

    def test_pred_token_ids(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.pred_token_ids == sample_token_ids[1:]

    def test_pred_logprobs(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.pred_logprobs == sample_logprobs[1:]

    def test_pred_logits(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
        )
        assert traj.pred_logits == sample_logits[1:]

    def test_pred_full_logits_none(self, sample_token_ids, sample_logprobs, sample_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
            full_logits=None,
        )
        assert traj.pred_full_logits is None

    def test_pred_full_logits_sliced(self, sample_token_ids, sample_logprobs, sample_logits, sample_full_logits):
        traj = TokenTrajectory(
            token_ids=sample_token_ids,
            logprobs=sample_logprobs,
            logits=sample_logits,
            full_logits=sample_full_logits,
        )
        assert traj.pred_full_logits.shape[0] == len(sample_token_ids) - 1


class TestTokenTrajectoryContinuationText:
    """Tests for continuation_text properties."""

    def test_continuation_text_with_prefill(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -1.5],
            logits=[0.0, 1.0, 1.5],
            prefill_text="Hello ",
            generated_text="world!",
        )
        assert traj.continuation_text == "Hello world!"

    def test_continuation_text_no_prefill(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -1.5],
            logits=[0.0, 1.0, 1.5],
            prefill_text=None,
            generated_text="world!",
        )
        assert traj.continuation_text == "world!"

    def test_continuation_text_no_generated(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -1.5],
            logits=[0.0, 1.0, 1.5],
            prefill_text="Hello",
            generated_text=None,
        )
        assert traj.continuation_text is None

    def test_continuation_text_no_thinking(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4],
            logprobs=[0.0, -1.0, -1.5, -2.0],
            logits=[0.0, 1.0, 1.5, 2.0],
            prefill_text="",
            generated_text="<think>Thinking...</think>Answer here",
        )
        assert traj.continuation_text_no_thinking == "Answer here"

    def test_continuation_text_no_thinking_without_blocks(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -1.5],
            logits=[0.0, 1.0, 1.5],
            prefill_text="",
            generated_text="No thinking blocks here",
        )
        assert traj.continuation_text_no_thinking == "No thinking blocks here"


class TestTokenTrajectoryTextAfterArm:
    """Tests for text_after_arm method."""

    def test_text_after_arm_basic(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4, 5],
            logprobs=[0.0, -1.0, -1.5, -2.0, -2.5],
            logits=[0.0, 1.0, 1.5, 2.0, 2.5],
            prefill_text="Hello ",
            generated_text="world here!",
            arm_text_lengths=[6],  # "Hello " is 6 chars
        )
        result = traj.text_after_arm(0)
        assert result == "world here!"

    def test_text_after_arm_no_arm_lengths(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -1.5],
            logits=[0.0, 1.0, 1.5],
            prefill_text="Hello ",
            generated_text="world!",
            arm_text_lengths=None,
        )
        # Should return full continuation
        result = traj.text_after_arm(0)
        assert result == "Hello world!"


class TestTokenTrajectoryGetConditionalProb:
    """Tests for get_conditional_prob method."""

    def test_get_conditional_prob_basic(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4],
            logprobs=[0.0, -1.0, -2.0, -0.5],
            logits=[0.0, 1.0, 2.0, 0.5],
        )
        # P(tokens 1-3) = exp(-1.0 + -2.0 + -0.5) = exp(-3.5)
        prob = traj.get_conditional_prob(1, 4)
        assert abs(prob - math.exp(-3.5)) < 1e-6

    def test_get_conditional_prob_single_token(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        prob = traj.get_conditional_prob(1, 2)
        assert abs(prob - math.exp(-1.0)) < 1e-6

    def test_get_conditional_prob_invalid_range(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        # Invalid: start >= end
        assert traj.get_conditional_prob(2, 1) is None

    def test_get_conditional_prob_out_of_bounds(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        # Invalid: end > length
        assert traj.get_conditional_prob(0, 10) is None

    def test_get_conditional_prob_negative_start(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        assert traj.get_conditional_prob(-1, 2) is None


class TestTokenTrajectorySanitize:
    """Tests for sanitize method."""

    def test_sanitize_replaces_nan(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, float("nan"), -1.0],
            logits=[0.0, float("nan"), 1.0],
        )
        traj.sanitize()
        assert not math.isnan(traj.logprobs[1])
        assert not math.isnan(traj.logits[1])

    def test_sanitize_replaces_inf(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, float("inf"), -1.0],
            logits=[0.0, float("-inf"), 1.0],
        )
        traj.sanitize()
        assert not math.isinf(traj.logprobs[1])
        assert not math.isinf(traj.logits[1])

    def test_sanitize_returns_self(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        result = traj.sanitize()
        assert result is traj


class TestTokenTrajectoryPopHeavy:
    """Tests for pop_heavy and pop_full_logits methods."""

    def test_pop_full_logits(self, sample_full_logits):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4, 5, 6],
            logprobs=[0.0, -1.0, -2.0, -1.5, -0.5, -1.0],
            logits=[0.0, 1.0, 2.0, 1.5, 0.5, 1.0],
            full_logits=sample_full_logits,
        )
        popped = traj.pop_full_logits()
        assert popped is not None
        assert traj.full_logits is None

    def test_pop_full_logits_when_none(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
            full_logits=None,
        )
        popped = traj.pop_full_logits()
        assert popped is None

    def test_pop_heavy_clears_full_logits(self, sample_full_logits):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4, 5, 6],
            logprobs=[0.0, -1.0, -2.0, -1.5, -0.5, -1.0],
            logits=[0.0, 1.0, 2.0, 1.5, 0.5, 1.0],
            full_logits=sample_full_logits,
        )
        traj.pop_heavy()
        assert traj.full_logits is None


class TestTokenTrajectoryToDict:
    """Tests for to_dict method."""

    def test_to_dict_excludes_full_logits(self, sample_full_logits):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3, 4, 5, 6],
            logprobs=[0.0, -1.0, -2.0, -1.5, -0.5, -1.0],
            logits=[0.0, 1.0, 2.0, 1.5, 0.5, 1.0],
            full_logits=sample_full_logits,
        )
        d = traj.to_dict()
        # full_logits should be None in dict (popped for serialization)
        assert d.get("full_logits") is None
        # But traj should still have it
        assert traj.full_logits is not None

    def test_to_dict_basic_fields(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
            prefill_text="Hello",
            generated_text="World",
        )
        d = traj.to_dict()
        assert d["token_ids"] == [1, 2, 3]
        assert d["prefill_text"] == "Hello"
        assert d["generated_text"] == "World"


class TestTokenTrajectoryInternals:
    """Tests for can_have_internals and has_internals methods."""

    def test_can_have_internals_false(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        assert traj.can_have_internals() is False

    def test_has_internals_false(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        assert traj.has_internals() is False

    def test_has_internals_for_false(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        assert traj.has_internals_for(lambda x: True) is False


class TestTokenTrajectoryFactoryFunction:
    """Tests using the sample trajectory factory."""

    def test_create_sample_trajectory(self):
        traj = create_sample_trajectory(n_tokens=15)
        assert traj.n_sequence == 15

    def test_create_sample_trajectory_with_arm_idx(self):
        traj = create_sample_trajectory(arm_idx=(1, 2))
        assert traj.arm_idx == (1, 2)

    def test_create_sample_trajectory_with_text(self):
        traj = create_sample_trajectory(
            prefill_text="Test prefix",
            generated_text="Test generated",
        )
        assert traj.prefill_text == "Test prefix"
        assert traj.generated_text == "Test generated"
