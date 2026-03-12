"""Seed utilities for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int | None) -> None:
    """Set random seeds for reproducibility across random, numpy, and torch."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
