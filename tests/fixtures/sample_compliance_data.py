"""Factory functions for creating sample compliance data for testing.

Provides utilities to create compliance vectors and scoring data
for estimation pipeline tests.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence


def create_compliance_vectors(
    n_samples: int = 10,
    n_structures: int = 4,
    seed: int = 42,
) -> list[list[float]]:
    """Create random compliance vectors for testing.

    Args:
        n_samples: Number of compliance vectors to generate
        n_structures: Number of structures (dimensions)
        seed: Random seed for reproducibility

    Returns:
        List of compliance vectors, each with n_structures elements in [0, 1]
    """
    rng = random.Random(seed)
    vectors = []
    for _ in range(n_samples):
        vec = [rng.random() for _ in range(n_structures)]
        vectors.append(vec)
    return vectors


def create_uniform_compliance_vectors(
    n_samples: int = 10,
    n_structures: int = 4,
) -> list[list[float]]:
    """Create compliance vectors with uniform values (0.5).

    Useful for testing neutral/balanced cases.

    Args:
        n_samples: Number of vectors to generate
        n_structures: Number of structures

    Returns:
        List of compliance vectors, all with value 0.5
    """
    return [[0.5] * n_structures for _ in range(n_samples)]


def create_extreme_compliance_vectors(
    n_samples: int = 10,
    n_structures: int = 4,
    pattern: str = "alternating",
) -> list[list[float]]:
    """Create compliance vectors with extreme values.

    Args:
        n_samples: Number of vectors
        n_structures: Number of structures
        pattern: "alternating" (0/1 alternating), "all_zero", "all_one"

    Returns:
        List of compliance vectors with extreme values
    """
    vectors = []
    for i in range(n_samples):
        if pattern == "alternating":
            vec = [float((j + i) % 2) for j in range(n_structures)]
        elif pattern == "all_zero":
            vec = [0.0] * n_structures
        elif pattern == "all_one":
            vec = [1.0] * n_structures
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
        vectors.append(vec)
    return vectors


def create_clustered_compliance_vectors(
    n_clusters: int = 2,
    samples_per_cluster: int = 5,
    n_structures: int = 4,
    seed: int = 42,
) -> tuple[list[list[float]], list[int]]:
    """Create compliance vectors clustered around different centers.

    Args:
        n_clusters: Number of clusters
        samples_per_cluster: Samples per cluster
        n_structures: Number of structures
        seed: Random seed

    Returns:
        Tuple of (compliance vectors, cluster labels)
    """
    rng = random.Random(seed)

    # Generate cluster centers
    centers = []
    for _ in range(n_clusters):
        center = [rng.random() for _ in range(n_structures)]
        centers.append(center)

    # Generate samples around centers
    vectors = []
    labels = []
    for cluster_idx, center in enumerate(centers):
        for _ in range(samples_per_cluster):
            # Add small noise around center
            vec = [
                max(0.0, min(1.0, c + rng.gauss(0, 0.1)))
                for c in center
            ]
            vectors.append(vec)
            labels.append(cluster_idx)

    return vectors, labels


@dataclass
class MockTrajectoryScoringData:
    """Mock trajectory scoring data for testing estimation pipeline."""

    structure_scores: list[float]
    log_prob: float
    n_tokens: int
    traj_idx: int
    branch: str = "trunk"


def create_trajectory_scoring_data(
    n_trajectories: int = 10,
    n_structures: int = 4,
    seed: int = 42,
) -> list[MockTrajectoryScoringData]:
    """Create mock TrajectoryScoringData objects for testing.

    Args:
        n_trajectories: Number of trajectories
        n_structures: Number of structures
        seed: Random seed

    Returns:
        List of MockTrajectoryScoringData objects
    """
    rng = random.Random(seed)

    data = []
    for i in range(n_trajectories):
        scores = [rng.random() for _ in range(n_structures)]
        log_prob = -rng.uniform(1.0, 10.0)  # Negative log prob
        n_tokens = rng.randint(10, 50)

        data.append(
            MockTrajectoryScoringData(
                structure_scores=scores,
                log_prob=log_prob,
                n_tokens=n_tokens,
                traj_idx=i,
                branch="trunk" if i < n_trajectories // 2 else "branch_1",
            )
        )

    return data


def create_weights_for_trajectories(
    n_trajectories: int,
    weight_type: str = "uniform",
    seed: int = 42,
) -> list[float]:
    """Create probability weights for trajectories.

    Args:
        n_trajectories: Number of trajectories
        weight_type: "uniform", "skewed", or "random"
        seed: Random seed

    Returns:
        List of weights summing to 1.0
    """
    if weight_type == "uniform":
        return [1.0 / n_trajectories] * n_trajectories

    rng = random.Random(seed)

    if weight_type == "skewed":
        # First trajectory gets most weight
        raw = [0.5] + [0.5 / (n_trajectories - 1)] * (n_trajectories - 1)
    elif weight_type == "random":
        raw = [rng.random() for _ in range(n_trajectories)]
    else:
        raise ValueError(f"Unknown weight_type: {weight_type}")

    # Normalize to sum to 1
    total = sum(raw)
    return [w / total for w in raw]
