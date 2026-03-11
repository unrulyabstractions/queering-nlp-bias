"""Core logging primitives."""

from __future__ import annotations

import sys


def log(msg: str = "", end: str = "\n", gap: int = 0) -> None:
    """Print with immediate flush.

    Args:
        msg: Message to print
        end: Line ending (default newline)
        gap: Number of blank lines to print before the message
    """
    for _ in range(gap):
        print(flush=True)
    print(msg, end=end, flush=True)


def log_flush() -> None:
    """Flush stdout."""
    sys.stdout.flush()
