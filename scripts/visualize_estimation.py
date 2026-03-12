"""Generate visualizations from an existing estimation output JSON.

Usage:
    python scripts/visualize_estimation.py out/<method>/est_<name>.json
    python scripts/visualize_estimation.py out/<method>/est_<name>.json --output-dir out/<method>/viz

The generation and scoring JSONs are auto-inferred from the estimation path
and the `generation_file` field embedded in the scoring output. Supply them
explicitly with --generation / --scoring if auto-inference fails.

Outputs:
    out/<method>/viz/...  (PNG plots)
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


def infer_method_from_path(est_path: Path) -> str:
    """Extract generation method name from estimation path.

    Path pattern: out/<method>/est_<trial>_<scoring>.json
    Example: out/simple-sampling/est_example_example.json -> simple-sampling
    """
    return est_path.parent.name


def infer_score_path(est_path: Path) -> Path:
    """Infer scoring JSON path from estimation path.

    out/<method>/est_<name>.json -> out/<method>/score_<name>.json
    """
    name = est_path.stem.replace("est_", "", 1)
    return est_path.parent / f"score_{name}.json"


def infer_gen_path_from_score(score_path: Path) -> Path | None:
    """Read generation_file field from the scoring JSON.

    Args:
        score_path: Path to the scoring output JSON file.

    Returns:
        Path to the generation file if found, None otherwise.
    """
    try:
        with open(score_path) as f:
            data = json.load(f)
        gen_file = data.get("generation_file", "")
        if gen_file:
            return Path(gen_file)
    except (OSError, json.JSONDecodeError, KeyError):
        # File not readable, invalid JSON, or missing expected structure
        return None
    return None


def resolve_paths(
    est_path: Path,
    gen_override: str | None,
    score_override: str | None,
) -> OutputPaths:
    """Resolve all three pipeline paths, auto-inferring where possible."""
    # Scoring path
    if score_override:
        score_path = Path(score_override)
    else:
        score_path = infer_score_path(est_path)
        if not score_path.exists():
            print(
                f"Warning: scoring file not found at {score_path} "
                "(dynamics plots will be skipped). Use --scoring to specify it.",
                file=sys.stderr,
            )
            score_path = est_path  # fallback — dynamics plot will silently skip

    # Generation path
    if gen_override:
        gen_path = Path(gen_override)
    else:
        gen_path = None
        if score_path.exists() and score_path != est_path:
            gen_path = infer_gen_path_from_score(score_path)
        if gen_path is None or not gen_path.exists():
            if gen_path is not None:
                print(
                    f"Warning: generation file not found at {gen_path} "
                    "(tree plots will be skipped). Use --generation to specify it.",
                    file=sys.stderr,
                )
            gen_path = est_path  # fallback — tree plots will silently skip

    return OutputPaths(generation=gen_path, judgment=score_path, estimation=est_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations from an estimation output JSON"
    )
    parser.add_argument(
        "estimation", help="Path to estimation output JSON (out/<method>/est_*.json)"
    )
    parser.add_argument(
        "--generation",
        help="Path to generation JSON (out/<method>/gen_*.json). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--scoring",
        help="Path to scoring JSON (out/<method>/score_*.json). Auto-inferred if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write plots into (default: out/<method>/viz)",
    )
    args = parser.parse_args()

    est_path = Path(args.estimation)
    if not est_path.exists():
        print(f"Error: file not found: {est_path}", file=sys.stderr)
        sys.exit(1)

    method = infer_method_from_path(est_path)
    paths = resolve_paths(est_path, args.generation, args.scoring)

    result = EstimationResult.from_estimation_file(method, paths)
    output_dir = Path(args.output_dir) if args.output_dir else None
    created = visualize_result(result, output_dir=output_dir)

    if created:
        actual_dir = args.output_dir if args.output_dir else f"out/{method}/viz"
        print(f"Saved {len(created)} plots to {actual_dir}/")
    else:
        print("No plots were generated.")


if __name__ == "__main__":
    main()
