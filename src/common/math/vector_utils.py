"""Vector math utilities for estimation computations.

Provides functions for computing orientation vectors, norms, and
related vector operations used in the estimation pipeline.
"""

from __future__ import annotations

import math


def compute_orientation_vector(
    source_core: list[float],
    reference_core: list[float] | None,
) -> tuple[list[float], float]:
    """Compute orientation vector and its norm from source to reference.

    Orientation = source_core - reference_core (vector difference).
    The norm is the Euclidean distance (L2 norm).

    Args:
        source_core: The source core vector (e.g., branch core)
        reference_core: The reference core vector (e.g., trunk core).
            If None or empty, returns empty vector and zero norm.

    Returns:
        Tuple of (orientation_vector, orientation_norm).
        If reference_core is None or either core is empty, returns ([], 0.0).
    """
    if reference_core is None or not source_core or not reference_core:
        return [], 0.0

    if len(source_core) != len(reference_core):
        raise ValueError(
            f"Core dimension mismatch: source has {len(source_core)}, "
            f"reference has {len(reference_core)}"
        )

    orientation = [source_core[i] - reference_core[i] for i in range(len(source_core))]
    norm = math.sqrt(sum(v * v for v in orientation))

    return orientation, norm
