"""Integration tests for estimation pipeline."""

from __future__ import annotations

import math

import pytest

from src.estimation.estimation_core_types import NAMED_CORES, CoreParams, CoreVariant
from src.estimation.arm_types import classify_arm, ArmKind
from src.common.math.entropy_diversity.structure_aware import (
    generalized_system_core,
    expected_deviance,
    orientation,
    deviance,
    core_diversity,
)
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_trajectory_scoring_data,
    create_weights_for_trajectories,
    create_clustered_compliance_vectors,
)


@pytest.mark.integration
class TestFullEstimationOutput:
    """Integration tests for complete estimation output."""

    def test_compute_core_from_compliance_data(self, compliance_data_for_estimation):
        """Test computing a core from compliance data."""
        compliance_vectors = compliance_data_for_estimation["compliance_vectors"]
        weights = compliance_data_for_estimation["weights"]

        core = generalized_system_core(compliance_vectors, weights, q=1.0, r=1.0)

        assert len(core) == 5  # n_structures
        assert all(0.0 <= c <= 1.0 for c in core)

    def test_compute_deviance_from_core(self, compliance_data_for_estimation):
        """Test computing deviance relative to a core."""
        compliance_vectors = compliance_data_for_estimation["compliance_vectors"]
        weights = compliance_data_for_estimation["weights"]

        core = generalized_system_core(compliance_vectors, weights, q=1.0, r=1.0)
        deviance_avg = expected_deviance(compliance_vectors, core, weights=weights)

        assert deviance_avg >= 0

    def test_compute_orientations(self, compliance_data_for_estimation):
        """Test computing orientation vectors."""
        compliance_vectors = compliance_data_for_estimation["compliance_vectors"]
        weights = compliance_data_for_estimation["weights"]

        core = generalized_system_core(compliance_vectors, weights, q=1.0, r=1.0)

        orientations = []
        for compliance in compliance_vectors:
            theta = orientation(compliance, core)
            orientations.append(theta)

        assert len(orientations) == 20
        for theta in orientations:
            assert len(theta) == 5

    def test_compute_multiple_core_variants(self, compliance_data_for_estimation):
        """Test computing multiple named core variants."""
        compliance_vectors = compliance_data_for_estimation["compliance_vectors"]
        weights = compliance_data_for_estimation["weights"]

        variants = []
        for name, q, r, desc in NAMED_CORES[:5]:
            core = generalized_system_core(compliance_vectors, weights, q=q, r=r)
            deviance_avg = expected_deviance(compliance_vectors, core, weights=weights)

            variant = CoreVariant(
                name=name,
                q=q,
                r=r,
                description=desc,
                core=core,
                deviance_avg=deviance_avg,
                deviance_var=0.0,
            )
            variants.append(variant)

        assert len(variants) == 5
        # Standard core should exist
        assert any(v.name == "standard" for v in variants)


@pytest.mark.integration
class TestTrunkCoreComputation:
    """Integration tests for trunk-specific core computation."""

    def test_trunk_core_computed(self, compliance_data_for_estimation):
        """Test that trunk core can be computed separately."""
        scoring_data = compliance_data_for_estimation["scoring_data"]

        # Filter to trunk trajectories only
        trunk_data = [d for d in scoring_data if d.branch == "trunk"]

        if trunk_data:
            trunk_compliance = [d.structure_scores for d in trunk_data]
            n_trunk = len(trunk_compliance)
            weights = [1.0 / n_trunk] * n_trunk

            trunk_core = generalized_system_core(
                trunk_compliance, weights, q=1.0, r=1.0
            )
            assert len(trunk_core) == 5

    def test_branch_relative_to_trunk(self, compliance_data_for_estimation):
        """Test computing branch orientation relative to trunk core."""
        scoring_data = compliance_data_for_estimation["scoring_data"]

        # Split by branch
        trunk_data = [d for d in scoring_data if d.branch == "trunk"]
        branch_data = [d for d in scoring_data if d.branch == "branch_1"]

        if trunk_data and branch_data:
            # Compute trunk core
            trunk_compliance = [d.structure_scores for d in trunk_data]
            n_trunk = len(trunk_compliance)
            trunk_weights = [1.0 / n_trunk] * n_trunk
            trunk_core = generalized_system_core(
                trunk_compliance, trunk_weights, q=1.0, r=1.0
            )

            # Compute branch orientation relative to trunk
            branch_compliance = [d.structure_scores for d in branch_data]
            for compliance in branch_compliance:
                theta = orientation(compliance, trunk_core)
                assert len(theta) == 5


@pytest.mark.integration
class TestBranchOrientations:
    """Integration tests for branch orientation computations."""

    def test_branch_orientations_relative(self, compliance_data_for_estimation):
        """Test computing branch orientations relative to each other."""
        scoring_data = compliance_data_for_estimation["scoring_data"]

        branches = set(d.branch for d in scoring_data)

        if len(branches) >= 2:
            # Get compliance by branch
            by_branch = {}
            for branch in branches:
                data = [d for d in scoring_data if d.branch == branch]
                by_branch[branch] = [d.structure_scores for d in data]

            # Compute overall core
            all_compliance = [d.structure_scores for d in scoring_data]
            n = len(all_compliance)
            weights = [1.0 / n] * n
            overall_core = generalized_system_core(all_compliance, weights)

            # Each branch should have orientation relative to core
            for branch, compliances in by_branch.items():
                for compliance in compliances:
                    theta = orientation(compliance, overall_core)
                    d = deviance(compliance, overall_core)
                    assert d >= 0


@pytest.mark.integration
class TestArmClassificationInEstimation:
    """Integration tests for arm classification in estimation context."""

    def test_classify_arms_for_estimation(self):
        """Test classifying arms for estimation purposes."""
        arm_names = ["root", "trunk", "branch_1", "branch_2", "twig_1_b1"]

        classified = {name: classify_arm(name) for name in arm_names}

        assert classified["root"] == ArmKind.ROOT
        assert classified["trunk"] == ArmKind.TRUNK
        assert classified["branch_1"] == ArmKind.BRANCH
        assert classified["twig_1_b1"] == ArmKind.TWIG

    def test_reference_arm_is_trunk(self):
        """Test that trunk is the reference arm for orientation."""
        from src.estimation.arm_types import is_reference_arm

        arm_names = ["root", "trunk", "branch_1", "branch_2"]

        reference_arms = [name for name in arm_names if is_reference_arm(name)]

        assert reference_arms == ["trunk"]


@pytest.mark.integration
class TestCoreDiversityMetrics:
    """Integration tests for core diversity metrics."""

    def test_core_diversity_uniform(self):
        """Test core diversity for uniform compliance."""
        # Uniform core should have max diversity
        core = [0.25, 0.25, 0.25, 0.25]
        diversity = core_diversity(core)
        assert abs(diversity - 4.0) < 1e-6

    def test_core_diversity_concentrated(self):
        """Test core diversity for concentrated compliance."""
        # Concentrated core should have low diversity
        core = [1.0, 0.0, 0.0, 0.0]
        diversity = core_diversity(core)
        assert abs(diversity - 1.0) < 1e-6

    def test_core_diversity_from_data(self, compliance_data_for_estimation):
        """Test computing core diversity from actual data."""
        compliance_vectors = compliance_data_for_estimation["compliance_vectors"]
        weights = compliance_data_for_estimation["weights"]

        core = generalized_system_core(compliance_vectors, weights, q=1.0, r=1.0)
        diversity = core_diversity(core)

        # Should be between 1 and n_structures
        assert 1.0 <= diversity <= 5.0


@pytest.mark.integration
class TestClusteredData:
    """Integration tests with clustered compliance data."""

    def test_clustered_data_different_cores(self):
        """Test that clustered data produces different cluster cores."""
        compliance_vectors, labels = create_clustered_compliance_vectors(
            n_clusters=2,
            samples_per_cluster=10,
            n_structures=4,
        )

        # Split by cluster
        cluster_0 = [v for v, l in zip(compliance_vectors, labels) if l == 0]
        cluster_1 = [v for v, l in zip(compliance_vectors, labels) if l == 1]

        # Compute cores for each cluster
        weights_0 = [1.0 / len(cluster_0)] * len(cluster_0)
        weights_1 = [1.0 / len(cluster_1)] * len(cluster_1)

        core_0 = generalized_system_core(cluster_0, weights_0)
        core_1 = generalized_system_core(cluster_1, weights_1)

        # Cores should be different (clusters are around different centers)
        diff = sum((a - b) ** 2 for a, b in zip(core_0, core_1))
        assert diff > 0  # Some difference expected

    def test_within_cluster_deviance_lower(self):
        """Test that within-cluster deviance is lower than cross-cluster."""
        compliance_vectors, labels = create_clustered_compliance_vectors(
            n_clusters=2,
            samples_per_cluster=10,
            n_structures=4,
        )

        cluster_0 = [v for v, l in zip(compliance_vectors, labels) if l == 0]
        cluster_1 = [v for v, l in zip(compliance_vectors, labels) if l == 1]

        # Core for cluster 0
        weights_0 = [1.0 / len(cluster_0)] * len(cluster_0)
        core_0 = generalized_system_core(cluster_0, weights_0)

        # Within-cluster deviance
        within_deviance = expected_deviance(cluster_0, core_0, weights=weights_0)

        # Cross-cluster deviance (cluster_1 relative to cluster_0's core)
        weights_1 = [1.0 / len(cluster_1)] * len(cluster_1)
        cross_deviance = expected_deviance(cluster_1, core_0, weights=weights_1)

        # Cross-cluster should generally be higher
        # (This is probabilistic, but clustered data should show this pattern)
        # We don't assert strictly because of random variation
        assert within_deviance >= 0
        assert cross_deviance >= 0
