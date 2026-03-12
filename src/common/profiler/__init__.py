"""Simple profiling utilities.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())

# Explicit import needed: 'profile' is excluded by auto_export (stdlib collision)
from .profiling_decorators import profile

__all__.append("profile")
