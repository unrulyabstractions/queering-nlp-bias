"""Core logging primitives for console output.

Provides the base log() function and simple progress/section utilities.
"""

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


def log_progress(current: int, total: int, prefix: str = "") -> None:
    """Print progress indicator (overwrites line)."""
    log(f"{prefix}{current}/{total}", end="\r")


def log_done(msg: str = "") -> None:
    """Print completion message (clears progress line)."""
    log(msg)


def log_section(title: str) -> None:
    """Print a section header."""
    log(f"\n{title}")
