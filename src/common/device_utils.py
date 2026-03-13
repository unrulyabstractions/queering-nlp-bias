"""Device and memory utilities for GPU/MPS/CPU operations."""

from __future__ import annotations

import gc
import os

import psutil
import torch

from src.common.logging import log


def get_device() -> str:
    """Return the best available device: cuda, mps, or cpu."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_memory_usage() -> dict:
    """Return current memory usage statistics for available accelerators and system RAM."""
    stats = {}

    # GPU memory
    if torch.cuda.is_available():
        stats["cuda_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
    if hasattr(torch.mps, "current_allocated_memory"):
        try:
            stats["mps_alloc_gb"] = torch.mps.current_allocated_memory() / 1e9
        except Exception:
            pass

    # System RAM (cross-platform)
    proc = psutil.Process(os.getpid())
    stats["ram_gb"] = proc.memory_info().rss / 1e9
    stats["ram_percent"] = proc.memory_percent()

    return stats


def log_memory(stage: str, verbose: bool = False) -> None:
    """Print memory usage at a given stage (verbose mode only)."""
    mem = get_memory_usage()
    if mem and verbose:
        mem_str = ", ".join(f"{k}={v:.2f}" for k, v in mem.items())
        log(f"  [Memory @ {stage}] {mem_str}")


def log_mem(label: str) -> None:
    """Log current memory usage with a label (always prints)."""
    mem = get_memory_usage()
    ram = mem.get("ram_gb", 0)
    mps = mem.get("mps_alloc_gb", 0)
    cuda = mem.get("cuda_alloc_gb", 0)
    accel = mps if mps > 0 else cuda
    log(f"  [{label}] RAM: {ram:.2f}GB, Accelerator: {accel:.2f}GB")


def clear_gpu_memory() -> None:
    """Clear GPU memory caches for CUDA and MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
