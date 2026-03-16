"""Console output utilities."""

from __future__ import annotations

from datetime import datetime


def log_timestamped(msg: str) -> None:
    """Print with timestamp prefix."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def log_section(title: str) -> None:
    """Print a section header."""
    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 60, flush=True)
