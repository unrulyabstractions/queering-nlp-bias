"""Tests for weighting methods in estimation pipeline."""

from __future__ import annotations

import math

import pytest

from fixtures.sample_compliance_data import (
    create_trajectory_scoring_data,
    create_weights_for_trajectories,
)


class TestUniformWeights:
    """Tests for uniform weighting."""

    def test_uniform_weights_equal(self):
        weights = create_weights_for_trajectories(5, weight_type="uniform")
        assert len(weights) == 5
        expected = 1.0 / 5
        for w in weights:
            assert abs(w - expected) < 1e-10

    def test_uniform_weights_sum_to_one(self):
        weights = create_weights_for_trajectories(10, weight_type="uniform")
        assert abs(sum(weights) - 1.0) < 1e-10


class TestRandomWeights:
    """Tests for random weighting."""

    def test_random_weights_sum_to_one(self):
        weights = create_weights_for_trajectories(10, weight_type="random")
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_random_weights_all_positive(self):
        weights = create_weights_for_trajectories(10, weight_type="random")
        assert all(w > 0 for w in weights)

    def test_random_weights_deterministic_with_seed(self):
        weights1 = create_weights_for_trajectories(10, weight_type="random", seed=42)
        weights2 = create_weights_for_trajectories(10, weight_type="random", seed=42)
        assert weights1 == weights2


class TestSkewedWeights:
    """Tests for skewed weighting."""

    def test_skewed_weights_first_largest(self):
        weights = create_weights_for_trajectories(5, weight_type="skewed")
        assert weights[0] == max(weights)

    def test_skewed_weights_sum_to_one(self):
        weights = create_weights_for_trajectories(5, weight_type="skewed")
        assert abs(sum(weights) - 1.0) < 1e-10


class TestWeightingIntegration:
    """Integration tests for weighting with scoring data."""

    def test_weights_match_trajectory_count(self):
        data = create_trajectory_scoring_data(n_trajectories=15)
        weights = create_weights_for_trajectories(15, weight_type="uniform")
        assert len(weights) == len(data)

    def test_weighted_mean_basic(self):
        # Simple weighted mean calculation
        values = [1.0, 2.0, 3.0, 4.0]
        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_mean = sum(v * w for v, w in zip(values, weights))
        expected = 1.0 * 0.4 + 2.0 * 0.3 + 3.0 * 0.2 + 4.0 * 0.1
        assert abs(weighted_mean - expected) < 1e-10

    def test_uniform_weighted_mean_equals_simple_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        weights = create_weights_for_trajectories(5, weight_type="uniform")
        weighted_mean = sum(v * w for v, w in zip(values, weights))
        simple_mean = sum(values) / len(values)
        assert abs(weighted_mean - simple_mean) < 1e-10
