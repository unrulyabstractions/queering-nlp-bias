"""Utilities for extracting continuation text from trajectory dicts.

Provides a consistent way to get the continuation text from trajectory
dictionaries, handling the fallback from stored continuation_text to
computed prefill_text + generated_text.
"""

from __future__ import annotations

from typing import Any


def get_continuation_text(traj: dict[str, Any]) -> str:
    """Get continuation text from trajectory dict.

    Computes from prefill_text + generated_text if continuation_text is not stored.

    Args:
        traj: Trajectory dictionary containing text fields

    Returns:
        The continuation text (prefill + generated text)
    """
    stored = traj.get("continuation_text")
    if stored:
        return stored
    prefill = traj.get("prefill_text") or ""
    generated = traj.get("generated_text") or ""
    return prefill + generated
