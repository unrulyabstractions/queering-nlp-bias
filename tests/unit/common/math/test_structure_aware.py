"""Tests for structure-aware diversity metrics."""

from __future__ import annotations

import math

import pytest
import torch

from src.common.math.entropy_diversity.structure_aware import (
    orientation,
    deviance,
    normalized_deviance,
    generalized_structure_core,
    generalized_system_core,
    expected_deviance,
    deviance_variance,
    expected_orientation,
    core_entropy,
    core_diversity,
    excess_deviance,
    deficit_deviance,
    mutual_deviance,
    expected_excess_deviance,
    expected_deficit_deviance,
    expected_mutual_deviance,
)


class TestOrientation:
    """Tests for orientation function."""

    def test_orientation_basic(self):
        compliance = [0.8, 0.3, 0.5]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        # Use pytest.approx for floating point comparison
        assert result[0] == pytest.approx(0.3)
        assert result[1] == pytest.approx(-0.2)
        assert result[2] == pytest.approx(0.0)

    def test_orientation_zero_when_equal(self):
        compliance = [0.5, 0.5, 0.5]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        assert all(abs(x) < 1e-10 for x in result)

    def test_orientation_positive_over_compliance(self):
        # Over-compliance: compliance > core
        compliance = [1.0, 1.0, 1.0]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        assert all(x > 0 for x in result)

    def test_orientation_negative_under_compliance(self):
        # Under-compliance: compliance < core
        compliance = [0.0, 0.0, 0.0]
        core = [0.5, 0.5, 0.5]
        result = orientation(compliance, core)
        assert all(x < 0 for x in result)

    def test_orientation_torch_tensor(self):
        compliance = torch.tensor([0.8, 0.3, 0.5])
        core = torch.tensor([0.5, 0.5, 0.5])
        result = orientation(compliance, core)
        expected = torch.tensor([0.3, -0.2, 0.0])
        assert torch.allclose(result, expected)

    def test_orientation_length_mismatch_raises(self):
        compliance = [0.8, 0.3]
        core = [0.5, 0.5, 0.5]
        with pytest.raises(ValueError):
            orientation(compliance, core)


class TestDeviance:
    """Tests for deviance function."""

    def test_deviance_l2_basic(self):
        compliance = [0.8, 0.3, 0.5]
        core = [0.5, 0.5, 0.5]
        result = deviance(compliance, core, norm="l2")
        # L2 norm of [0.3, -0.2, 0.0] = sqrt(0.09 + 0.04) = sqrt(0.13)
        expected = math.sqrt(0.3**2 + 0.2**2)
        assert abs(result - expected) < 1e-6

    def test_deviance_l1(self):
        compliance = [0.8, 0.3, 0.5]
        core = [0.5, 0.5, 0.5]
        result = deviance(compliance, core, norm="l1")
        # L1 norm of [0.3, -0.2, 0.0] = 0.5
        expected = 0.3 + 0.2
        assert abs(result - expected) < 1e-6

    def test_deviance_linf(self):
        compliance = [0.8, 0.3, 0.5]
        core = [0.5, 0.5, 0.5]
        result = deviance(compliance, core, norm="linf")
        # Linf norm of [0.3, -0.2, 0.0] = 0.3
        assert abs(result - 0.3) < 1e-6

    def test_deviance_zero_when_equal(self):
        compliance = [0.5, 0.5, 0.5]
        core = [0.5, 0.5, 0.5]
        result = deviance(compliance, core)
        assert abs(result) < 1e-10

    def test_deviance_empty_input(self):
        result = deviance([], [])
        assert result == 0.0

    def test_deviance_torch_tensor(self):
        compliance = torch.tensor([0.8, 0.3])
        core = torch.tensor([0.5, 0.5])
        result = deviance(compliance, core, norm="l2")
        expected = torch.tensor(0.3**2 + 0.2**2).sqrt()
        assert torch.isclose(result, expected)


class TestNormalizedDeviance:
    """Tests for normalized_deviance function."""

    def test_normalized_deviance_in_01(self):
        compliance = [0.8, 0.3, 0.5, 0.9]
        core = [0.5, 0.5, 0.5, 0.5]
        result = normalized_deviance(compliance, core, norm="l2")
        assert 0.0 <= result <= 1.0

    def test_normalized_deviance_max_is_one(self):
        # Maximum deviance: all components differ by 1
        compliance = [1.0, 1.0, 1.0, 1.0]
        core = [0.0, 0.0, 0.0, 0.0]
        result = normalized_deviance(compliance, core, norm="l2")
        assert abs(result - 1.0) < 1e-6


class TestGeneralizedStructureCore:
    """Tests for generalized_structure_core function."""

    def test_standard_core_q1_r1(self):
        # q=1, r=1 should be standard expected value
        compliances = [0.2, 0.4, 0.6, 0.8]
        probs = [0.25, 0.25, 0.25, 0.25]  # Uniform
        result = generalized_structure_core(compliances, probs, q=1.0, r=1.0)
        expected = 0.5  # Mean of 0.2, 0.4, 0.6, 0.8
        assert abs(result - expected) < 1e-6

    def test_empty_input(self):
        result = generalized_structure_core([], [], q=1.0, r=1.0)
        assert result == 0.0


class TestGeneralizedSystemCore:
    """Tests for generalized_system_core function."""

    def test_system_core_basic(self):
        compliances = [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
        ]
        probs = [1/3, 1/3, 1/3]
        result = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        assert len(result) == 2
        # First structure: mean of [0.2, 0.4, 0.6] = 0.4
        # Second structure: mean of [0.8, 0.6, 0.4] = 0.6
        assert abs(result[0] - 0.4) < 1e-6
        assert abs(result[1] - 0.6) < 1e-6

    def test_system_core_empty(self):
        result = generalized_system_core([], [])
        assert result == []


class TestExpectedDeviance:
    """Tests for expected_deviance function."""

    def test_expected_deviance_basic(self):
        compliances = [
            [0.8, 0.3],
            [0.2, 0.7],
        ]
        core = [0.5, 0.5]
        result = expected_deviance(compliances, core)
        assert result > 0

    def test_expected_deviance_with_weights(self):
        compliances = [
            [0.8, 0.3],
            [0.2, 0.7],
        ]
        core = [0.5, 0.5]
        weights = [0.9, 0.1]  # First sample weighted heavily
        result = expected_deviance(compliances, core, weights=weights)
        # Should be close to deviance of first sample
        first_deviance = deviance([0.8, 0.3], core)
        assert abs(result - first_deviance) < 0.1  # Approximate

    def test_expected_deviance_empty(self):
        result = expected_deviance([], [0.5, 0.5])
        assert result == 0.0

    def test_expected_deviance_uniform_weights(self):
        compliances = [[0.8, 0.3], [0.2, 0.7]]
        core = [0.5, 0.5]
        # No weights should use uniform
        result = expected_deviance(compliances, core, weights=None)
        assert result > 0


class TestDevianceVariance:
    """Tests for deviance_variance function."""

    def test_deviance_variance_basic(self):
        compliances = [
            [0.8, 0.3],
            [0.2, 0.7],
            [0.5, 0.5],
        ]
        core = [0.5, 0.5]
        result = deviance_variance(compliances, core)
        assert result >= 0

    def test_deviance_variance_all_same(self):
        # All same compliance -> zero variance
        compliances = [[0.5, 0.5]] * 5
        core = [0.5, 0.5]
        result = deviance_variance(compliances, core)
        assert abs(result) < 1e-10


class TestExpectedOrientation:
    """Tests for expected_orientation function."""

    def test_expected_orientation_basic(self):
        compliances = [
            [0.8, 0.2],
            [0.2, 0.8],
        ]
        core = [0.5, 0.5]
        result = expected_orientation(compliances, core)
        # Mean of [0.3, -0.3] and [-0.3, 0.3] = [0, 0]
        assert len(result) == 2
        assert abs(result[0]) < 1e-6
        assert abs(result[1]) < 1e-6

    def test_expected_orientation_empty(self):
        result = expected_orientation([], [0.5, 0.5])
        assert result == []


class TestCoreStatistics:
    """Tests for core entropy and diversity."""

    def test_core_entropy_uniform(self):
        # Uniform core should have max entropy
        core = [0.25, 0.25, 0.25, 0.25]
        result = core_entropy(core)
        expected = math.log(4)  # ln(4) for 4 equally weighted
        assert abs(result - expected) < 1e-6

    def test_core_entropy_concentrated(self):
        # Concentrated core should have low entropy
        core = [1.0, 0.0, 0.0, 0.0]
        result = core_entropy(core)
        assert result < 0.1  # Should be close to 0

    def test_core_diversity_uniform(self):
        # Uniform -> effective number = actual number
        core = [0.25, 0.25, 0.25, 0.25]
        result = core_diversity(core)
        assert abs(result - 4.0) < 1e-6

    def test_core_diversity_concentrated(self):
        # Concentrated -> effective number = 1
        core = [1.0, 0.0, 0.0, 0.0]
        result = core_diversity(core)
        assert abs(result - 1.0) < 1e-6


class TestRelativeEntropyDeviance:
    """Tests for excess, deficit, and mutual deviance."""

    def test_excess_deviance_basic(self):
        compliance = [0.8, 0.2]
        core = [0.5, 0.5]
        result = excess_deviance(compliance, core)
        assert result >= 1.0  # Deviance is always >= 1

    def test_deficit_deviance_basic(self):
        compliance = [0.8, 0.2]
        core = [0.5, 0.5]
        result = deficit_deviance(compliance, core)
        assert result >= 1.0

    def test_mutual_deviance_basic(self):
        compliance = [0.8, 0.2]
        core = [0.5, 0.5]
        result = mutual_deviance(compliance, core)
        assert 1.0 <= result <= 2.0  # JSD-based, bounded

    def test_mutual_deviance_symmetric(self):
        compliance = [0.8, 0.2]
        core = [0.5, 0.5]
        result1 = mutual_deviance(compliance, core)
        result2 = mutual_deviance(core, compliance)
        assert abs(result1 - result2) < 1e-6

    def test_excess_deviance_equals_one_when_match(self):
        # When compliance matches core, deviance should be 1
        compliance = [0.5, 0.5]
        core = [0.5, 0.5]
        result = excess_deviance(compliance, core)
        assert abs(result - 1.0) < 1e-6


class TestExpectedRelativeDeviance:
    """Tests for expected excess/deficit/mutual deviance."""

    def test_expected_excess_deviance_basic(self):
        compliances = [[0.8, 0.2], [0.6, 0.4]]
        core = [0.5, 0.5]
        result = expected_excess_deviance(compliances, core)
        assert result >= 1.0

    def test_expected_deficit_deviance_basic(self):
        compliances = [[0.8, 0.2], [0.6, 0.4]]
        core = [0.5, 0.5]
        result = expected_deficit_deviance(compliances, core)
        assert result >= 1.0

    def test_expected_mutual_deviance_basic(self):
        compliances = [[0.8, 0.2], [0.6, 0.4]]
        core = [0.5, 0.5]
        result = expected_mutual_deviance(compliances, core)
        assert 1.0 <= result <= 2.0

    def test_expected_mutual_deviance_empty(self):
        result = expected_mutual_deviance([], [0.5, 0.5])
        assert result == 1.0  # Neutral value
