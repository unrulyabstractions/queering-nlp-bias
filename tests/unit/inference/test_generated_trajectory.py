"""Tests for GeneratedTrajectory class."""

from __future__ import annotations

import math

import pytest
import torch

from src.inference.generated_trajectory import (
    GeneratedTrajectory,
    calculate_trajectories_for_batch,
)
from src.common.token_trajectory import TokenTrajectory


class TestGeneratedTrajectoryFromInference:
    """Tests for GeneratedTrajectory.from_inference class method."""

    def test_from_inference_basic(self):
        token_ids = [1, 100, 200, 300]
        logits = torch.randn(4, 1000)  # [n_sequence, vocab_size]
        traj = GeneratedTrajectory.from_inference(token_ids, logits, device="cpu")

        assert traj.token_ids == token_ids
        assert len(traj.logprobs) == 4
        assert len(traj.logits) == 4
        assert traj.full_logits is not None

    def test_from_inference_first_token_logprob_zero(self):
        token_ids = [1, 100, 200]
        logits = torch.randn(3, 1000)
        traj = GeneratedTrajectory.from_inference(token_ids, logits)

        # First token has logprob 0 (it's given)
        assert traj.logprobs[0] == 0.0
        assert traj.logits[0] == 0.0

    def test_from_inference_logprobs_negative(self):
        token_ids = [1, 100, 200, 300]
        logits = torch.randn(4, 1000)
        traj = GeneratedTrajectory.from_inference(token_ids, logits)

        # All prediction logprobs should be negative (log of probability < 1)
        for lp in traj.pred_logprobs:
            assert lp <= 0

    def test_from_inference_with_internals(self):
        token_ids = [1, 50]  # Token IDs must be < vocab_size (100)
        logits = torch.randn(2, 100)
        internals = {"layer_5": torch.randn(10, 10)}
        traj = GeneratedTrajectory.from_inference(
            token_ids, logits, internals=internals
        )

        assert "layer_5" in traj.internals

    def test_from_inference_full_logits_shape(self):
        token_ids = [1, 100, 200, 300, 400]
        vocab_size = 500
        logits = torch.randn(5, vocab_size)
        traj = GeneratedTrajectory.from_inference(token_ids, logits)

        # full_logits should have same shape as input, offset by 1
        assert traj.full_logits.shape == (5, vocab_size)


class TestGeneratedTrajectoryFromLogprobs:
    """Tests for GeneratedTrajectory.from_logprobs class method."""

    def test_from_logprobs_basic(self):
        token_ids = [1, 100, 200, 300]
        logprobs = [0.0, -1.0, -2.0, -1.5]
        traj = GeneratedTrajectory.from_logprobs(token_ids, logprobs)

        assert traj.token_ids == token_ids
        assert traj.logprobs == logprobs
        assert traj.full_logits is None  # No full logits from logprobs-only

    def test_from_logprobs_logits_equal_logprobs(self):
        token_ids = [1, 100]
        logprobs = [0.0, -1.5]
        traj = GeneratedTrajectory.from_logprobs(token_ids, logprobs)

        # Logits should equal logprobs (scalar approximation)
        assert traj.logits == logprobs


class TestGeneratedTrajectoryFromTokenTrajectory:
    """Tests for GeneratedTrajectory.from_token_trajectory class method."""

    def test_from_token_trajectory_basic(self):
        base = TokenTrajectory(
            token_ids=[1, 100, 200],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        traj = GeneratedTrajectory.from_token_trajectory(base)

        assert traj.token_ids == base.token_ids
        assert traj.logprobs == base.logprobs
        assert isinstance(traj, GeneratedTrajectory)

    def test_from_token_trajectory_with_internals(self):
        base = TokenTrajectory(
            token_ids=[1, 100],
            logprobs=[0.0, -1.0],
            logits=[0.0, 1.0],
        )
        internals = {"test_key": torch.randn(5)}
        traj = GeneratedTrajectory.from_token_trajectory(base, internals=internals)

        assert "test_key" in traj.internals


class TestGeneratedTrajectoryInternals:
    """Tests for internals handling."""

    def test_can_have_internals(self):
        traj = GeneratedTrajectory(
            token_ids=[1],
            logprobs=[0.0],
            logits=[0.0],
        )
        assert traj.can_have_internals() is True

    def test_has_internals_empty(self):
        traj = GeneratedTrajectory(
            token_ids=[1],
            logprobs=[0.0],
            logits=[0.0],
            internals={},
        )
        assert traj.has_internals() is False

    def test_has_internals_with_data(self):
        traj = GeneratedTrajectory(
            token_ids=[1],
            logprobs=[0.0],
            logits=[0.0],
            internals={"some_key": torch.randn(5)},
        )
        assert traj.has_internals() is True

    def test_has_internals_for_with_filter(self):
        traj = GeneratedTrajectory(
            token_ids=[1],
            logprobs=[0.0],
            logits=[0.0],
            internals={"layer_5_attn": torch.randn(5), "other": torch.randn(3)},
        )
        assert traj.has_internals_for(lambda x: "layer" in x) is True
        assert traj.has_internals_for(lambda x: "nonexistent" in x) is False

    def test_pop_heavy_clears_internals(self):
        traj = GeneratedTrajectory(
            token_ids=[1],
            logprobs=[0.0],
            logits=[0.0],
            internals={"data": torch.randn(100)},
            full_logits=torch.randn(1, 100),
        )
        traj.pop_heavy()
        assert traj.internals == {}
        assert traj.full_logits is None


class TestCalculateTrajectoriesForBatch:
    """Tests for calculate_trajectories_for_batch function."""

    def test_batch_basic(self):
        token_ids_batch = [
            [1, 10, 20],  # Token IDs must be < vocab_size (100)
            [1, 15, 25, 35],
        ]
        # Padded logits: max length is 4
        logits_batch = torch.randn(2, 4, 100)
        trajs = calculate_trajectories_for_batch(token_ids_batch, logits_batch)

        assert len(trajs) == 2
        assert trajs[0].n_sequence == 3
        assert trajs[1].n_sequence == 4

    def test_batch_trims_padding(self):
        token_ids_batch = [
            [1, 10],  # Length 2, token IDs < vocab_size (100)
            [1, 15, 25, 35, 45],  # Length 5
        ]
        # Padded to max length 5
        logits_batch = torch.randn(2, 5, 100)
        trajs = calculate_trajectories_for_batch(token_ids_batch, logits_batch)

        # Each trajectory should only have logits for actual tokens
        assert trajs[0].full_logits.shape[0] == 2
        assert trajs[1].full_logits.shape[0] == 5

    def test_batch_single_sequence(self):
        token_ids_batch = [[1, 10, 20, 30]]  # Token IDs < vocab_size (50)
        logits_batch = torch.randn(1, 4, 50)
        trajs = calculate_trajectories_for_batch(token_ids_batch, logits_batch)

        assert len(trajs) == 1
        assert isinstance(trajs[0], GeneratedTrajectory)


class TestGeneratedTrajectoryProperties:
    """Tests for inherited properties from TokenTrajectory."""

    def test_n_sequence(self):
        traj = GeneratedTrajectory(
            token_ids=[1, 100, 200, 300],
            logprobs=[0.0, -1.0, -2.0, -1.5],
            logits=[0.0, 1.0, 2.0, 1.5],
        )
        assert traj.n_sequence == 4

    def test_pred_token_ids(self):
        traj = GeneratedTrajectory(
            token_ids=[1, 100, 200],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        assert traj.pred_token_ids == [100, 200]

    def test_continuation_text(self):
        traj = GeneratedTrajectory(
            token_ids=[1, 100],
            logprobs=[0.0, -1.0],
            logits=[0.0, 1.0],
            prefill_text="Hello ",
            generated_text="world!",
        )
        assert traj.continuation_text == "Hello world!"
