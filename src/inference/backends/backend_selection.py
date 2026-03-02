"""Recommended backend selection based on use case and hardware.

Usage:
    from src.inference.backends import get_recommended_backend_inference

    # Pure inference (fastest generation)
    backend = get_recommended_backend_inference()
"""

from __future__ import annotations

import torch

from .model_backend import ModelBackend


def _is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return torch.backends.mps.is_available()


def _mlx_available() -> bool:
    """Check if MLX is installed."""
    try:
        import mlx.core

        return True
    except ImportError:
        return False


def get_recommended_backend_inference() -> ModelBackend:
    """Get the recommended backend for pure inference (generation/logprobs).

    Prioritizes speed. MLX is fastest on Apple Silicon, HuggingFace
    is most compatible and second-fastest on all platforms.

    Returns:
        ModelBackend: Recommended backend for inference
    """
    if _is_apple_silicon() and _mlx_available():
        return ModelBackend.MLX
    return ModelBackend.HUGGINGFACE
