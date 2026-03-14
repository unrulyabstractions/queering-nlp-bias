#!/usr/bin/env python3
"""Regenerate visualizations for experiments.

An experiment trial is a directory containing:
    generation.json (in parent or current)
    scoring.json
    estimation.json

Usage:
    # All experiments in out/
    uv run python scripts/visualize_experiment.py

    # Filter by name
    uv run python scripts/visualize_experiment.py example

    # Publication quality
    uv run python scripts/visualize_experiment.py --camera-ready
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.experiment_types import OutputPaths
from src.common.logging import log, log_header
from src.estimation.estimation_experiment_types import EstimationResult
from src.viz import visualize_result


def get_experiment_trial_dirs(
    base_dir: Path, name_filter: str | None = None
) -> list[Path]:
    """Find experiment trial directories (folders containing estimation.json).

    Args:
        base_dir: Root directory to search
        name_filter: If set, filter to trials whose path contains this string

    Returns:
        Sorted list of experiment trial directories
    """
    trial_dirs = []
    for estimation_file in base_dir.rglob("estimation.json"):
        trial_dir = estimation_file.parent
        if name_filter and name_filter not in str(trial_dir):
            continue
        trial_dirs.append(trial_dir)
    return sorted(trial_dirs)


def resolve_experiment_paths(trial_dir: Path) -> OutputPaths:
    """Resolve all paths for an experiment trial.

    Expected structure:
        <base>/<gen_name>/generation.json
        <base>/<gen_name>/<scoring_name>/scoring.json
        <base>/<gen_name>/<scoring_name>/estimation.json

    Args:
        trial_dir: Directory containing estimation.json

    Returns:
        OutputPaths with generation, judgment, and estimation paths
    """
    gen_dir = trial_dir.parent
    return OutputPaths(
        generation=gen_dir / "generation.json",
        judgment=trial_dir / "scoring.json",
        estimation=trial_dir / "estimation.json",
    )


def visualize_trial(trial_dir: Path, *, camera_ready: bool = False) -> list[Path]:
    """Visualize a single experiment trial.

    Args:
        trial_dir: Directory containing estimation.json
        camera_ready: If True, use high DPI (300) for publication quality

    Returns:
        List of created visualization files
    """
    paths = resolve_experiment_paths(trial_dir)
    method_name = paths.generation.parent.name

    try:
        result = EstimationResult.from_estimation_file(method_name, paths)
        return visualize_result(result, camera_ready=camera_ready)
    except Exception as e:
        log(f"  Error: {e}")
        return []


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Regenerate visualizations for experiments"
    )
    parser.add_argument(
        "filter",
        nargs="?",
        default=None,
        help="Filter experiments by name (e.g., 'example')",
    )
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Base directory to search (default: out)",
    )
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Publication quality: high DPI (300), all annotations enabled",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.out_dir)

    if not base_dir.exists():
        print(f"Error: directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    trial_dirs = get_experiment_trial_dirs(base_dir, args.filter)

    if not trial_dirs:
        filter_msg = f" matching '{args.filter}'" if args.filter else ""
        print(f"No experiment trials found in {base_dir}/{filter_msg}")
        sys.exit(0)

    log_header(f"Visualizing {len(trial_dirs)} experiment(s)")

    total_plots = 0
    for trial_dir in trial_dirs:
        rel_path = trial_dir.relative_to(base_dir)
        log(f"\n[{rel_path}]")

        created = visualize_trial(trial_dir, camera_ready=args.camera_ready)
        if created:
            log(f"  Created {len(created)} plots")
            total_plots += len(created)

    log(f"\nTotal: {total_plots} plots generated")


if __name__ == "__main__":
    main()
