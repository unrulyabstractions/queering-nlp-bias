"""Estimation weighting methods.

Each weighting method defines how to convert (log_probs, n_tokens) into
normalized weights for computing cores, deviances, and orientations.

To DISABLE a method, set ENABLED = False at the top of its file.
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
