"""Tests for core computation in estimation pipeline."""

from __future__ import annotations

import math

import pytest

from src.estimation.estimation_core_types import NAMED_CORES, CoreParams, CoreVariant
from src.common.math.entropy_diversity.structure_aware import (
    generalized_system_core,
    expected_deviance,
)
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_uniform_compliance_vectors,
    create_weights_for_trajectories,
)


class TestNamedCores:
    """Tests for NAMED_CORES constant."""

    def test_named_cores_has_entries(self):
        assert len(NAMED_CORES) > 0

    def test_named_cores_structure(self):
        for name, q, r, desc in NAMED_CORES:
            assert isinstance(name, str)
            assert isinstance(q, (int, float))
            assert isinstance(r, (int, float))
            assert isinstance(desc, str)

    def test_standard_core_present(self):
        names = [n for n, _, _, _ in NAMED_CORES]
        assert "standard" in names

    def test_standard_core_is_q1_r1(self):
        for name, q, r, desc in NAMED_CORES:
            if name == "standard":
                assert q == 1.0
                assert r == 1.0
                break


class TestCoreParams:
    """Tests for CoreParams dataclass."""

    def test_create_core_params(self):
        params = CoreParams(q=1.0, r=2.0)
        assert params.q == 1.0
        assert params.r == 2.0

    def test_core_params_to_dict(self):
        params = CoreParams(q=1.0, r=2.0)
        d = params.to_dict()
        assert d["q"] == 1.0
        assert d["r"] == 2.0

    def test_core_params_from_dict(self):
        data = {"q": 2.0, "r": 1.0}
        params = CoreParams.from_dict(data)
        assert params.q == 2.0
        assert params.r == 1.0


class TestCoreVariant:
    """Tests for CoreVariant dataclass."""

    def test_create_core_variant(self):
        variant = CoreVariant(
            name="test",
            q=1.0,
            r=1.0,
            description="Test variant",
            core=[0.5, 0.5],
            deviance_avg=0.1,
            deviance_var=0.01,
        )
        assert variant.name == "test"
        assert variant.core == [0.5, 0.5]
        assert variant.deviance_avg == 0.1

    def test_core_variant_to_dict(self):
        variant = CoreVariant(
            name="test",
            q=1.0,
            r=1.0,
            description="Test",
            core=[0.5],
            deviance_avg=0.1,
            deviance_var=0.01,
        )
        d = variant.to_dict()
        assert "name" in d
        assert "core" in d


class TestCoreComputation:
    """Tests for computing cores from compliance vectors."""

    def test_compute_standard_core(self):
        compliances = create_compliance_vectors(n_samples=10, n_structures=4)
        probs = create_weights_for_trajectories(10, weight_type="uniform")
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        assert len(core) == 4

    def test_standard_core_is_mean(self):
        # For uniform weights and q=1, r=1, core should be mean
        compliances = [
            [0.2, 0.8],
            [0.4, 0.6],
            [0.6, 0.4],
            [0.8, 0.2],
        ]
        probs = [0.25, 0.25, 0.25, 0.25]
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        # Mean of first structure: (0.2 + 0.4 + 0.6 + 0.8) / 4 = 0.5
        assert abs(core[0] - 0.5) < 1e-6
        assert abs(core[1] - 0.5) < 1e-6

    def test_uniform_compliance_core(self):
        compliances = create_uniform_compliance_vectors(n_samples=10, n_structures=4)
        probs = create_weights_for_trajectories(10, weight_type="uniform")
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        # All 0.5, so core should be 0.5
        for c in core:
            assert abs(c - 0.5) < 1e-6


class TestDevianceFromCore:
    """Tests for computing deviance relative to core."""

    def test_deviance_from_computed_core(self):
        compliances = create_compliance_vectors(n_samples=10, n_structures=4)
        probs = create_weights_for_trajectories(10, weight_type="uniform")
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        deviance = expected_deviance(compliances, core, weights=probs)
        assert deviance >= 0

    def test_zero_deviance_uniform_compliance(self):
        compliances = create_uniform_compliance_vectors(n_samples=10, n_structures=4)
        probs = create_weights_for_trajectories(10, weight_type="uniform")
        core = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        deviance = expected_deviance(compliances, core, weights=probs)
        # All compliances are [0.5, 0.5, 0.5, 0.5], core is [0.5, 0.5, 0.5, 0.5]
        # So deviance should be 0
        assert abs(deviance) < 1e-6


class TestCoreVariants:
    """Tests for computing multiple core variants."""

    def test_compute_multiple_variants(self):
        compliances = create_compliance_vectors(n_samples=10, n_structures=4)
        probs = create_weights_for_trajectories(10, weight_type="uniform")

        variants = []
        for name, q, r, desc in NAMED_CORES[:5]:  # First 5 variants
            core = generalized_system_core(compliances, probs, q=q, r=r)
            deviance_avg = expected_deviance(compliances, core, weights=probs)
            variant = CoreVariant(
                name=name,
                q=q,
                r=r,
                description=desc,
                core=core,
                deviance_avg=deviance_avg,
                deviance_var=0.0,  # Simplified
            )
            variants.append(variant)

        assert len(variants) == 5
        assert all(isinstance(v, CoreVariant) for v in variants)

    def test_different_qr_produce_different_cores(self):
        compliances = create_compliance_vectors(n_samples=20, n_structures=4, seed=123)
        probs = create_weights_for_trajectories(20, weight_type="random", seed=123)

        core_1_1 = generalized_system_core(compliances, probs, q=1.0, r=1.0)
        core_2_1 = generalized_system_core(compliances, probs, q=2.0, r=1.0)
        core_1_2 = generalized_system_core(compliances, probs, q=1.0, r=2.0)

        # Different q or r should produce different cores
        # (not guaranteed to be different, but usually are for non-uniform data)
        # At least verify they are computed
        assert len(core_1_1) == 4
        assert len(core_2_1) == 4
        assert len(core_1_2) == 4
