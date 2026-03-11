"""Common callback type aliases used across the codebase.

This module provides standardized callback types for logging,
progress reporting, and other cross-cutting concerns.
"""

from __future__ import annotations

from typing import Callable

# Logging callback: receives a message string
LogFn = Callable[[str], None]

# Progress callback: receives (name, current, total)
ProgressFn = Callable[[str, int, int], None]
