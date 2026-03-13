"""Tests for probability utility functions."""

from __future__ import annotations

import math

import pytest

from src.common.math.probability_utils import (
    normalize_log_probs,
    normalize_indexed_log_probs,
    compute_inv_perplexity_weights,
)


class TestNormalizeLogProbs:
    """Tests for normalize_log_probs function."""

    def test_normalize_basic(self):
        log_probs = [-1.0, -2.0, -3.0]
        result = normalize_log_probs(log_probs)
        assert len(result) == 3
        # Should sum to 1
        assert abs(sum(result) - 1.0) < 1e-6

    def test_normalize_empty_list(self):
        result = normalize_log_probs([])
        assert result == []

    def test_normalize_single_element(self):
        result = normalize_log_probs([-5.0])
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6  # Should be 1.0

    def test_normalize_all_equal(self):
        log_probs = [-1.0, -1.0, -1.0, -1.0]
        result = normalize_log_probs(log_probs)
        # Should be uniform
        expected = 0.25
        for p in result:
            assert abs(p - expected) < 1e-6

    def test_normalize_preserves_relative_order(self):
        # Higher log_prob should result in higher probability
        log_probs = [-1.0, -2.0, -3.0]
        result = normalize_log_probs(log_probs)
        assert result[0] > result[1] > result[2]

    def test_normalize_all_neg_inf_returns_uniform(self):
        log_probs = [float("-inf"), float("-inf"), float("-inf")]
        result = normalize_log_probs(log_probs)
        # Should be uniform
        expected = 1.0 / 3
        for p in result:
            assert abs(p - expected) < 1e-6

    def test_normalize_with_one_neg_inf(self):
        log_probs = [-1.0, float("-inf"), -2.0]
        result = normalize_log_probs(log_probs)
        # The -inf should have probability ~0
        assert result[1] < 1e-10
        # Others should sum to ~1
        assert abs(result[0] + result[2] - 1.0) < 1e-6

    def test_normalize_numerical_stability(self):
        # Very large negative values should still work
        log_probs = [-1000.0, -1001.0, -1002.0]
        result = normalize_log_probs(log_probs)
        assert abs(sum(result) - 1.0) < 1e-6


class TestNormalizeIndexedLogProbs:
    """Tests for normalize_indexed_log_probs function."""

    def test_normalize_indexed_basic(self):
        indexed = [(0, -1.0), (1, -2.0)]
        result = normalize_indexed_log_probs(indexed)
        assert len(result) == 2
        # Should preserve indices
        assert result[0][0] in [0, 1]
        assert result[1][0] in [0, 1]

    def test_normalize_indexed_empty(self):
        result = normalize_indexed_log_probs([])
        assert result == []

    def test_normalize_indexed_sorted_descending(self):
        indexed = [(0, -1.0), (1, -3.0), (2, -2.0)]
        result = normalize_indexed_log_probs(indexed, descending=True)
        # Sorted by probability descending
        assert result[0][1] >= result[1][1] >= result[2][1]

    def test_normalize_indexed_not_sorted(self):
        indexed = [(0, -1.0), (1, -2.0)]
        result = normalize_indexed_log_probs(indexed, descending=False)
        # Should preserve original order
        assert result[0][0] == 0
        assert result[1][0] == 1

    def test_normalize_indexed_sums_to_one(self):
        indexed = [(0, -1.0), (1, -2.0), (2, -1.5)]
        result = normalize_indexed_log_probs(indexed)
        total = sum(p for _, p in result)
        assert abs(total - 1.0) < 1e-6


class TestComputeInvPerplexityWeights:
    """Tests for compute_inv_perplexity_weights function."""

    def test_inv_ppl_basic(self):
        log_probs = [-10.0, -20.0]
        n_tokens = [10, 20]
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6

    def test_inv_ppl_equal_per_token_prob(self):
        # Same per-token probability should give equal weights
        log_probs = [-10.0, -20.0]  # Total log probs
        n_tokens = [10, 20]  # Per-token: -1.0 each
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        # Should be approximately equal
        assert abs(result[0] - result[1]) < 1e-6

    def test_inv_ppl_empty(self):
        result = compute_inv_perplexity_weights([], [])
        assert result == []

    def test_inv_ppl_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            compute_inv_perplexity_weights([-1.0, -2.0], [10])

    def test_inv_ppl_with_zero_tokens(self):
        # Zero tokens should be handled gracefully
        log_probs = [-10.0, -20.0]
        n_tokens = [10, 0]  # Second has 0 tokens
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        assert len(result) == 2

    def test_inv_ppl_very_negative_logprob(self):
        # Extremely negative log probs (underflow prevention)
        log_probs = [-1000.0, -10.0]
        n_tokens = [100, 10]
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        # Should still sum to 1 and handle gracefully
        assert abs(sum(result) - 1.0) < 1e-6

    def test_inv_ppl_all_underflow_returns_uniform(self):
        # All very negative -> should return uniform
        log_probs = [-1000.0, -1000.0]
        n_tokens = [1, 1]
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        # Should be uniform
        assert abs(result[0] - 0.5) < 1e-6
        assert abs(result[1] - 0.5) < 1e-6


class TestNormalizationProperties:
    """Property-based tests for normalization functions."""

    def test_normalize_output_in_01(self):
        log_probs = [-1.0, -5.0, -0.5, -10.0]
        result = normalize_log_probs(log_probs)
        for p in result:
            assert 0.0 <= p <= 1.0

    def test_normalize_max_element_largest(self):
        log_probs = [-1.0, -0.1, -5.0]  # -0.1 is highest
        result = normalize_log_probs(log_probs)
        assert result[1] == max(result)

    def test_normalize_min_element_smallest(self):
        log_probs = [-1.0, -0.1, -5.0]  # -5.0 is lowest
        result = normalize_log_probs(log_probs)
        assert result[2] == min(result)
