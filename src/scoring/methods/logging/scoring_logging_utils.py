"""Logging utilities for scoring methods.

This module provides logging functions for detailed scoring output.
"""

from __future__ import annotations

from src.common.callback_types import LogFn
from src.common.logging import oneline
from src.common.viz_utils import preview
from src.scoring.scoring_data import TrajectoryData


def log_trajectory_header(
    traj: TrajectoryData,
    idx: int,
    total: int,
    log_fn: LogFn,
    selected_text: str | None = None,
) -> None:
    """Log trajectory header with selected text.

    Args:
        traj: Trajectory being scored
        idx: Current trajectory index (0-based)
        total: Total number of trajectories
        log_fn: Logging callback
        selected_text: The actual text being scored (after string_selection applied)
    """
    text = selected_text if selected_text is not None else traj.continuation_text
    log_fn(f"Trajectory {idx + 1}/{total} (arm: {traj.arm_name})")
    log_fn(f'  Selected: "{preview(oneline(text), 120)}"')


def log_scoring_section(section_name: str, log_fn: LogFn) -> None:
    """Log a scoring section header.

    Args:
        section_name: Name of the section (e.g., "Categorical", "Graded", "Similarity")
        log_fn: Logging callback
    """
    log_fn(f"  {section_name}:")


def log_parse_failure(
    method_name: str,
    question: str,
    raw_response: str,
    log_fn: LogFn,
) -> None:
    """Log a parse failure warning for LLM-based scoring methods.

    Args:
        method_name: Name of the scoring method (e.g., "CATEGORICAL", "GRADED")
        question: The question that was asked
        raw_response: The raw response from the model
        log_fn: Logging callback
    """
    log_fn("")
    log_fn("  +----------------------------------------------------------------+")
    log_fn(f"  |  WARNING: {method_name} SCORE PARSE FAILURE - DEFAULTING TO 0.0")
    log_fn("  +----------------------------------------------------------------+")
    question_preview = question[:60] + "..." if len(question) > 60 else question
    log_fn(f"  Question: {question_preview}")
    log_fn(f"  Raw response: {repr(raw_response)}")
    log_fn("")
