"""Model backends module.

DO NOT add explicit __all__ lists here - use auto_export instead.
See src/common/auto_export.py for documentation on how this works.

Usage:
    # Get backend enum
    from src.inference.backends import ModelBackend

    # Get recommended backend for inference
    from src.inference.backends import get_recommended_backend_inference
"""

from src.common.auto_export import auto_export

__all__ = auto_export(__file__, __name__, globals())
