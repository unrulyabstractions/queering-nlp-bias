#!/usr/bin/env python3
"""Generate visualizations from an estimation JSON file.

This script is for visualizing legacy or non-standard experiment outputs.
For standard experiment directories, use visualize_experiment.py instead.

Usage:
    uv run python scripts/visualize_estimation.py path/to/estimation.json
    uv run python scripts/visualize_estimation.py path/to/estimation.json --camera-ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.common.experiment_types import OutputPaths
from src.estimation.estimation_experiment_types import EstimationResult
from src.viz import visualize_result


def read_json_field(path: Path, *keys: str) -> str | None:
    """Read a nested field from a JSON file."""
    try:
        with open(path) as f:
            data = json.load(f)
        for key in keys:
            data = data.get(key, {})
        return data if isinstance(data, str) else None
    except (OSError, json.JSONDecodeError):
        return None


def infer_generation_path(est_path: Path, score_path: Path | None) -> Path:
    """Infer generation.json path from scoring or estimation metadata."""
    # Try scoring file first
    if score_path and score_path.exists():
        gen_file = read_json_field(score_path, "generation_file")
        if gen_file:
            gen_path = Path(gen_file)
            if gen_path.exists():
                return gen_path

    # Try estimation metadata
    gen_file = read_json_field(est_path, "metadata", "generation_file")
    if gen_file:
        gen_path = Path(gen_file)
        if gen_path.exists():
            return gen_path
        print(f"Warning: generation file not found at {gen_path}", file=sys.stderr)

    return est_path  # Fallback - tree plots will skip


def build_paths(est_path: Path) -> OutputPaths:
    """Build OutputPaths by inferring related files."""
    # Try multiple scoring path patterns
    score_path = None
    candidates = [
        est_path.parent / "scoring.json",  # Standard name
        est_path.parent / f"score_{est_path.stem.replace('est_', '', 1)}.json",  # est_X -> score_X
    ]
    for candidate in candidates:
        if candidate.exists():
            score_path = candidate
            break

    if not score_path:
        print(f"Warning: scoring file not found in {est_path.parent}", file=sys.stderr)
        score_path = est_path  # Fallback

    # Try multiple generation path patterns
    gen_path = None
    gen_candidates = [
        est_path.parent.parent / "generation.json",  # Standard location
    ]
    for candidate in gen_candidates:
        if candidate.exists():
            gen_path = candidate
            break

    if not gen_path:
        gen_path = infer_generation_path(est_path, score_path if score_path != est_path else None)

    return OutputPaths(generation=gen_path, judgment=score_path, estimation=est_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from an estimation JSON"
    )
    parser.add_argument("estimation", help="Path to estimation.json")
    parser.add_argument("--output-dir", help="Output directory for plots")
    parser.add_argument(
        "--camera-ready",
        action="store_true",
        help="Publication quality: high DPI (300), all annotations enabled",
    )
    args = parser.parse_args()

    est_path = Path(args.estimation)
    if not est_path.exists():
        print(f"Error: file not found: {est_path}", file=sys.stderr)
        sys.exit(1)

    paths = build_paths(est_path)
    method_name = est_path.parent.name
    output_dir = Path(args.output_dir) if args.output_dir else None

    result = EstimationResult.from_estimation_file(method_name, paths)
    created = visualize_result(result, output_dir=output_dir, camera_ready=args.camera_ready)

    if created:
        print(f"Saved {len(created)} plots")
    else:
        print("No plots were generated.")


if __name__ == "__main__":
    main()
