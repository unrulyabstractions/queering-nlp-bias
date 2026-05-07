"""Logit-transformed KDE for densities supported on (0,1).

Standard Gaussian KDE on bounded data leaks mass past the interval and
underestimates density near the boundaries. The fix is to do KDE on the
logit-transformed data, then transform the density back via the
change-of-variables formula:

    f_Y(y) = f_Z(logit(y)) / (y * (1 - y))

The mode of Y is **not** sigmoid(mode of Z): the Jacobian shifts the
argmax. We must search for the argmax in y-space.

Public API:
- ``logit_kde_evaluate(samples, y_grid)`` -> density values f_Y at each grid point
- ``logit_kde_mode(samples)`` -> the mode of f_Y
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import gaussian_kde

# Numerical clipping: values exactly at 0 or 1 cannot be logit-transformed.
# We clip to (EPS, 1 - EPS) before transforming.
_EPS = 1e-6


def _to_array(samples: Sequence[float]) -> np.ndarray:
    """Convert to array, dropping NaN/inf."""
    arr = np.asarray(list(samples), dtype=float)
    return arr[np.isfinite(arr)]


def _logit(y: np.ndarray) -> np.ndarray:
    y = np.clip(y, _EPS, 1.0 - _EPS)
    return np.log(y / (1.0 - y))


def logit_kde_evaluate(
    samples: Sequence[float],
    y_grid: Sequence[float] | np.ndarray,
    *,
    bw_method: str | float = "silverman",
) -> np.ndarray:
    """Evaluate the logit-transformed KDE density f_Y on a y-grid.

    Args:
        samples: Sample values in [0, 1].
        y_grid: Grid of y values in (0, 1) where the density is evaluated.
        bw_method: Bandwidth selector passed to scipy's gaussian_kde
            ('silverman', 'scott', or a float).

    Returns:
        Array of density values f_Y at each grid point. Returns zeros if
        the sample is degenerate (fewer than 2 distinct interior values).
    """
    y = _to_array(samples)
    grid = np.clip(np.asarray(y_grid, dtype=float), _EPS, 1.0 - _EPS)

    if y.size < 2 or np.unique(y).size < 2:
        return np.zeros_like(grid)

    z = _logit(y)
    if np.unique(z).size < 2:
        return np.zeros_like(grid)

    kde = gaussian_kde(z, bw_method=bw_method)
    fz = kde(_logit(grid))
    return fz / (grid * (1.0 - grid))


def logit_kde_mode(
    samples: Sequence[float],
    *,
    bw_method: str | float = "silverman",
    n_grid: int = 1001,
) -> float:
    """Estimate the mode of f_Y via grid search on the logit-KDE.

    Returns the y in (0, 1) maximizing f_Y(y) = f_Z(logit y) / (y(1 - y)).
    Falls back to the sample mean when the sample is degenerate (all-equal,
    fewer than 2 interior points).

    Args:
        samples: Sample values in [0, 1].
        bw_method: Bandwidth selector passed to scipy's gaussian_kde.
        n_grid: Number of evaluation points on the y-grid.

    Returns:
        The estimated mode in (0, 1), or the sample mean as a fallback.
    """
    y = _to_array(samples)
    if y.size == 0:
        return 0.0

    if np.unique(y).size < 2:
        return float(y[0])

    grid = np.linspace(_EPS, 1.0 - _EPS, n_grid)
    fy = logit_kde_evaluate(y, grid, bw_method=bw_method)
    if not np.any(fy > 0):
        return float(np.mean(y))

    return float(grid[int(np.argmax(fy))])
