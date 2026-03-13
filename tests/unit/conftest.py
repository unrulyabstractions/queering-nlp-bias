"""Unit test fixtures."""

from __future__ import annotations

import pytest

from fixtures.mock_model_runner import MockModelRunner, create_mock_runner
from fixtures.sample_trajectories import (
    create_sample_trajectory,
    create_divergent_trajectories,
    create_trajectory_batch,
)
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_trajectory_scoring_data,
    create_weights_for_trajectories,
)


@pytest.fixture
def mock_runner() -> MockModelRunner:
    """Create a mock model runner for testing."""
    return create_mock_runner()


@pytest.fixture
def basic_trajectory():
    """Create a basic trajectory for testing."""
    return create_sample_trajectory()


@pytest.fixture
def divergent_trajs():
    """Create divergent trajectories for tree building tests."""
    return create_divergent_trajectories(n_trajectories=3, diverge_at=5)


@pytest.fixture
def trajectory_batch():
    """Create a batch of trajectories with groups."""
    return create_trajectory_batch(n_groups=2, trajs_per_group=3)


@pytest.fixture
def compliance_data():
    """Create compliance vectors for estimation tests."""
    return create_compliance_vectors(n_samples=10, n_structures=4)


@pytest.fixture
def scoring_data():
    """Create trajectory scoring data for estimation tests."""
    return create_trajectory_scoring_data(n_trajectories=10, n_structures=4)
