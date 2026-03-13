"""Integration test fixtures."""

from __future__ import annotations

import pytest

from fixtures.mock_model_runner import MockModelRunner, MockGenerationResponse
from fixtures.sample_trajectories import (
    create_trajectory_batch,
    create_divergent_trajectories,
)
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_trajectory_scoring_data,
    create_weights_for_trajectories,
)


@pytest.fixture
def integration_mock_runner() -> MockModelRunner:
    """Create a mock runner with pre-configured responses for integration tests."""
    runner = MockModelRunner()

    # Add some pre-configured responses for common test scenarios
    for i in range(10):
        runner.add_response(
            token_ids=[1, 100 + i, 200 + i, 300 + i],
            logprobs=[0.0, -1.0 - i * 0.1, -1.5, -0.8],
            generated_text=f"Generated response {i}",
        )

    return runner


@pytest.fixture
def large_trajectory_batch():
    """Create a larger batch of trajectories for integration testing."""
    return create_trajectory_batch(
        n_groups=3,
        trajs_per_group=5,
        shared_length=10,
        total_length=25,
    )


@pytest.fixture
def compliance_data_for_estimation():
    """Create compliance data suitable for full estimation pipeline tests."""
    n_trajectories = 20
    n_structures = 5
    return {
        "compliance_vectors": create_compliance_vectors(
            n_samples=n_trajectories,
            n_structures=n_structures,
        ),
        "scoring_data": create_trajectory_scoring_data(
            n_trajectories=n_trajectories,
            n_structures=n_structures,
        ),
        "weights": create_weights_for_trajectories(
            n_trajectories=n_trajectories,
            weight_type="random",
        ),
    }
