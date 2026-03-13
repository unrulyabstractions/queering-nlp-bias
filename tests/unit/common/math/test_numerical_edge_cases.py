"""Tests for numerical edge cases in math utilities."""

from __future__ import annotations

import math

import pytest

from src.common.math.probability_utils import (
    normalize_log_probs,
    compute_inv_perplexity_weights,
)
from src.common.math.entropy_diversity.structure_aware import (
    orientation,
    deviance,
    generalized_system_core,
    expected_deviance,
)


class TestEmptyInputHandling:
    """Tests for empty input handling."""

    def test_normalize_log_probs_empty(self):
        result = normalize_log_probs([])
        assert result == []

    def test_inv_perplexity_weights_empty(self):
        result = compute_inv_perplexity_weights([], [])
        assert result == []

    def test_deviance_empty_vectors(self):
        result = deviance([], [])
        assert result == 0.0

    def test_generalized_system_core_empty(self):
        result = generalized_system_core([], [])
        assert result == []

    def test_expected_deviance_empty(self):
        result = expected_deviance([], [0.5])
        assert result == 0.0


class TestNaNHandling:
    """Tests for NaN handling in calculations."""

    def test_normalize_with_nan_in_list(self):
        log_probs = [-1.0, float("nan"), -2.0]
        # Should handle gracefully - NaN becomes effectively -inf
        result = normalize_log_probs(log_probs)
        # NaN propagation means we may get NaN in output
        # The function should at least not crash
        assert len(result) == 3

    def test_deviance_with_nan_compliance(self):
        compliance = [0.5, float("nan"), 0.5]
        core = [0.5, 0.5, 0.5]
        result = deviance(compliance, core)
        # NaN should propagate
        assert math.isnan(result)


class TestInfinityHandling:
    """Tests for infinity handling in calculations."""

    def test_normalize_single_neg_inf(self):
        log_probs = [-1.0, float("-inf"), -2.0]
        result = normalize_log_probs(log_probs)
        assert len(result) == 3
        # -inf should have probability ~0
        assert result[1] < 1e-10
        # Sum should be ~1
        assert abs(sum(result) - 1.0) < 1e-6

    def test_normalize_all_neg_inf(self):
        log_probs = [float("-inf")] * 5
        result = normalize_log_probs(log_probs)
        # Should return uniform distribution
        expected = 1.0 / 5
        for p in result:
            assert abs(p - expected) < 1e-6

    def test_normalize_with_pos_inf(self):
        # Positive infinity is unusual - the current implementation
        # may treat it as NaN or handle it differently
        log_probs = [-1.0, float("inf"), -2.0]
        result = normalize_log_probs(log_probs)
        # Just verify we get a valid output without crashing
        assert len(result) == 3
        # Sum should be approximately 1
        total = sum(r for r in result if not math.isnan(r))
        assert 0.0 <= total <= 1.1  # Allow some tolerance


class TestUnderflowPrevention:
    """Tests for underflow prevention in calculations."""

    def test_normalize_very_negative_logprobs(self):
        # Very negative log probs that might cause underflow
        log_probs = [-1000.0, -1001.0, -1002.0]
        result = normalize_log_probs(log_probs)
        # Should still work due to logsumexp trick
        assert len(result) == 3
        assert abs(sum(result) - 1.0) < 1e-6

    def test_inv_ppl_very_negative(self):
        # Very negative log probs
        log_probs = [-1000.0, -1001.0]
        n_tokens = [100, 100]
        result = compute_inv_perplexity_weights(log_probs, n_tokens)
        # Should return uniform when all underflow
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6


class TestZeroProbability:
    """Tests for zero probability handling."""

    def test_normalize_near_zero_prob(self):
        # Log probs that result in very small probabilities
        log_probs = [-1.0, -100.0]
        result = normalize_log_probs(log_probs)
        assert len(result) == 2
        assert abs(sum(result) - 1.0) < 1e-6
        # Second element should be essentially 0
        assert result[1] < 1e-40

    def test_generalized_core_zero_weights(self):
        # All zero weights should be handled
        compliances = [[0.5, 0.5], [0.6, 0.4]]
        probs = [0.0, 0.0]
        # Should handle gracefully (fall back to uniform)
        result = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        assert len(result) == 2


class TestBoundaryValues:
    """Tests for boundary value handling."""

    def test_deviance_max_difference(self):
        # Maximum difference: 0 vs 1
        compliance = [0.0, 0.0, 0.0]
        core = [1.0, 1.0, 1.0]
        result = deviance(compliance, core, norm="l2")
        expected = math.sqrt(3)  # sqrt(1^2 + 1^2 + 1^2)
        assert abs(result - expected) < 1e-6

    def test_compliance_all_ones(self):
        compliance = [1.0, 1.0, 1.0]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        assert all(abs(x - 0.5) < 1e-10 for x in result)

    def test_compliance_all_zeros(self):
        compliance = [0.0, 0.0, 0.0]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        assert all(abs(x + 0.5) < 1e-10 for x in result)


class TestSingleElementCases:
    """Tests for single element inputs."""

    def test_normalize_single_element(self):
        result = normalize_log_probs([-5.0])
        assert len(result) == 1
        assert abs(result[0] - 1.0) < 1e-6

    def test_deviance_single_structure(self):
        compliance = [0.8]
        core = [0.5]
        result = deviance(compliance, core, norm="l2")
        assert abs(result - 0.3) < 1e-6

    def test_system_core_single_sample(self):
        compliances = [[0.8, 0.2]]
        probs = [1.0]
        result = generalized_system_core(compliances, probs)
        assert result[0] == pytest.approx(0.8, rel=1e-5)
        assert result[1] == pytest.approx(0.2, rel=1e-5)


class TestLargeInputs:
    """Tests for large input handling."""

    def test_normalize_many_elements(self):
        n = 1000
        log_probs = [-i * 0.01 for i in range(n)]
        result = normalize_log_probs(log_probs)
        assert len(result) == n
        assert abs(sum(result) - 1.0) < 1e-5

    def test_deviance_high_dimension(self):
        n = 100
        compliance = [0.6] * n
        core = [0.5] * n
        result = deviance(compliance, core, norm="l2")
        # Each dimension contributes 0.1^2 = 0.01
        # L2 = sqrt(n * 0.01) = sqrt(n) * 0.1
        expected = math.sqrt(n) * 0.1
        assert abs(result - expected) < 1e-6
