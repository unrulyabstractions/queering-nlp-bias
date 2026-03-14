"""Centralized output path computation for all pipeline stages.

This module defines the canonical directory structure for experiment outputs.
All path computation should go through these functions to ensure consistency.

Directory Structure:
    Normal mode (single method):
        out/<gen_name>/generation.json
        out/<gen_name>/<scoring_name>/scoring.json
        out/<gen_name>/<scoring_name>/estimation.json

    Multi-method mode (--all):
        generation_compare/<method>/<gen_name>/generation.json
        generation_compare/<method>/<gen_name>/<scoring_name>/scoring.json
        generation_compare/<method>/<gen_name>/<scoring_name>/estimation.json
"""

from __future__ import annotations

from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION PATHS
# ══════════════════════════════════════════════════════════════════════════════


def generation_output_path(
    config_path: Path,
    base_dir: str | Path = "out",
    method: str | None = None,
) -> Path:
    """Compute generation output path.

    Args:
        config_path: Path to generation config (used to get gen_name)
        base_dir: Output base directory
        method: If provided, include method in path (for multi-method mode)

    Returns:
        Path to generation.json
    """
    gen_name = config_path.stem
    base = Path(base_dir)
    if method:
        return base / method / gen_name / "generation.json"
    return base / gen_name / "generation.json"


def generation_summary_path(
    config_path: Path,
    base_dir: str | Path = "out",
    method: str | None = None,
) -> Path:
    """Compute generation summary path.

    Args:
        config_path: Path to generation config (used to get gen_name)
        base_dir: Output base directory
        method: If provided, include method in path (for multi-method mode)

    Returns:
        Path to summary_generation.txt
    """
    gen_name = config_path.stem
    base = Path(base_dir)
    if method:
        return base / method / gen_name / "summary_generation.txt"
    return base / gen_name / "summary_generation.txt"


# ══════════════════════════════════════════════════════════════════════════════
# SCORING PATHS
# ══════════════════════════════════════════════════════════════════════════════


def scoring_output_path(gen_output_path: Path, scoring_config_path: Path) -> Path:
    """Compute scoring output path (relative to generation directory).

    Args:
        gen_output_path: Path to generation.json
        scoring_config_path: Path to scoring config (used to get scoring_name)

    Returns:
        Path to scoring.json
    """
    gen_dir = gen_output_path.parent
    scoring_name = scoring_config_path.stem
    return gen_dir / scoring_name / "scoring.json"


def scoring_summary_path(gen_output_path: Path, scoring_config_path: Path) -> Path:
    """Compute scoring summary path (relative to generation directory).

    Args:
        gen_output_path: Path to generation.json
        scoring_config_path: Path to scoring config (used to get scoring_name)

    Returns:
        Path to summary_scoring.txt
    """
    gen_dir = gen_output_path.parent
    scoring_name = scoring_config_path.stem
    return gen_dir / scoring_name / "summary_scoring.txt"


# ══════════════════════════════════════════════════════════════════════════════
# ESTIMATION PATHS
# ══════════════════════════════════════════════════════════════════════════════


def estimation_output_path(scoring_output_path: Path) -> Path:
    """Compute estimation output path (same directory as scoring).

    Args:
        scoring_output_path: Path to scoring.json

    Returns:
        Path to estimation.json
    """
    return scoring_output_path.parent / "estimation.json"


def estimation_summary_path(scoring_output_path: Path) -> Path:
    """Compute estimation summary path (same directory as scoring).

    Args:
        scoring_output_path: Path to scoring.json

    Returns:
        Path to summary_estimation.txt
    """
    return scoring_output_path.parent / "summary_estimation.txt"


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION PATHS
# ══════════════════════════════════════════════════════════════════════════════


def viz_output_dir(estimation_path: Path) -> Path:
    """Compute visualization output directory.

    Args:
        estimation_path: Path to estimation.json

    Returns:
        Path to viz/ directory
    """
    return estimation_path.parent / "viz"


def dynamics_output_dir(estimation_path: Path) -> Path:
    """Compute dynamics visualization output directory.

    Args:
        estimation_path: Path to estimation.json

    Returns:
        Path to viz/dynamics/ directory
    """
    return estimation_path.parent / "viz" / "dynamics"
