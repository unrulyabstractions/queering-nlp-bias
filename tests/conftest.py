"""Root conftest.py for pytest fixtures and markers.

This module provides shared fixtures and marker registration for the test suite.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# Add tests directory to path for fixture imports
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))


# ══════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def sample_token_ids() -> list[int]:
    """Sample token ID sequence for testing."""
    return [1, 100, 200, 300, 400, 500]


@pytest.fixture
def sample_logprobs() -> list[float]:
    """Sample log probabilities matching sample_token_ids."""
    # First token has logprob 0 (given), rest have realistic negative values
    return [0.0, -1.5, -0.8, -2.1, -0.5, -1.2]


@pytest.fixture
def sample_logits() -> list[float]:
    """Sample logits matching sample_token_ids."""
    return [0.0, 5.2, 3.1, 2.8, 4.5, 3.9]


@pytest.fixture
def sample_full_logits() -> torch.Tensor:
    """Sample full logits tensor [n_sequence, vocab_size]."""
    vocab_size = 1000
    n_sequence = 6
    # Create random logits with reasonable values
    logits = torch.randn(n_sequence, vocab_size) * 2.0
    return logits


@pytest.fixture
def sample_weights() -> list[float]:
    """Sample probability weights for testing."""
    return [0.3, 0.4, 0.2, 0.1]


@pytest.fixture
def sample_compliance_vector() -> list[float]:
    """Sample system compliance vector."""
    return [0.8, 0.3, 0.5, 0.9]


@pytest.fixture
def sample_core() -> list[float]:
    """Sample system core (expected compliance)."""
    return [0.5, 0.5, 0.5, 0.5]


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def assert_floats_close(a: float, b: float, rel_tol: float = 1e-6) -> None:
    """Assert two floats are close, handling special values."""
    if math.isnan(a) and math.isnan(b):
        return
    if math.isinf(a) and math.isinf(b):
        assert (a > 0) == (b > 0), f"Infinities have different signs: {a} vs {b}"
        return
    assert math.isclose(a, b, rel_tol=rel_tol), f"{a} != {b} (rel_tol={rel_tol})"


def assert_lists_close(
    a: list[float], b: list[float], rel_tol: float = 1e-6
) -> None:
    """Assert two lists of floats are element-wise close."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert_floats_close(x, y, rel_tol), f"Mismatch at index {i}: {x} vs {y}"


def assert_sums_to_one(probs: list[float], tol: float = 1e-6) -> None:
    """Assert probabilities sum to 1.0."""
    total = sum(probs)
    assert abs(total - 1.0) < tol, f"Sum is {total}, expected 1.0"
