"""Generate trajectories using any registered generation method.

Unified script that supports all generation methods via --method argument.

Usage:
    uv run python scripts/generate_trajectories.py trials/generation/<config>.json
    uv run python scripts/generate_trajectories.py trials/generation/<config>.json \
        --method forking-paths
    uv run python scripts/generate_trajectories.py trials/generation/<config>.json \
        --method seeking-entropy --samples-per-expansion 5

Methods:
    simple-sampling (default): Temperature sampling with N samples per arm
    forking-paths: Systematic one-step deviations from greedy path
    seeking-entropy: Expand tree at high-uncertainty positions

Outputs:
    out/<method>/gen_<config>.json
    out/<method>/summary_gen_<config>.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas.script_utils import (
    apply_cli_overrides_to_config,
    load_model,
    log_prompt_header,
    run_generation_script,
)

from src.common.logging import log_section
from src.common.random_seed import set_seed
from src.generation import GenerationConfig, GenerationOutput, run_generation_pipeline
from src.generation.generation_method_registry import (
    get_output_name,
    get_params_class,
    list_output_names,
)
from src.generation.methods.logging import log_tree_trajectories


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for unified generation script."""
    available_methods = list_output_names()

    parser = argparse.ArgumentParser(
        description="Generate trajectories using any registered method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Methods:
  simple-sampling   Temperature sampling (default)
  forking-paths     Explore one-step deviations from greedy path
  seeking-entropy   Expand tree at high-entropy positions

Examples:
  %(prog)s config.json
  %(prog)s config.json --method forking-paths
  %(prog)s config.json --method seeking-entropy --samples-per-expansion 5
  %(prog)s config.json --samples-per-arm 20
""",
    )

    # Positional argument
    parser.add_argument("config", help="Path to generation config JSON")

    # Method selection
    parser.add_argument(
        "--method",
        choices=available_methods,
        default="simple-sampling",
        help=f"Generation method (default: simple-sampling). Choices: {', '.join(available_methods)}",
    )

    # Simple sampling arguments
    parser.add_argument(
        "--samples-per-arm",
        type=int,
        metavar="N",
        help="[simple-sampling] Trajectories per arm",
    )

    # Forking paths arguments
    parser.add_argument(
        "--max-alternates-per-position",
        type=int,
        metavar="K",
        help="[forking-paths] Max alternate tokens per position",
    )
    parser.add_argument(
        "--min-prob-for-alternate",
        type=float,
        metavar="P",
        help="[forking-paths] Minimum probability for alternate token",
    )
    parser.add_argument(
        "--min-entropy-to-fork",
        type=float,
        metavar="H",
        help="[forking-paths] Minimum entropy to consider forking",
    )
    parser.add_argument(
        "--samples-per-fork",
        type=int,
        metavar="N",
        help="[forking-paths] Continuations per fork point",
    )

    # Entropy seeking arguments
    parser.add_argument(
        "--samples-per-expansion",
        type=int,
        metavar="N",
        help="[seeking-entropy] Trajectories per expansion",
    )
    parser.add_argument(
        "--num-expansion-rounds",
        type=int,
        metavar="K",
        help="[seeking-entropy] Number of expansion rounds",
    )

    return parser.parse_args()


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Collect CLI argument overrides into a dict."""
    return {
        "samples_per_arm": args.samples_per_arm,
        "max_alternates_per_position": args.max_alternates_per_position,
        "min_prob_for_alternate": args.min_prob_for_alternate,
        "min_entropy_to_fork": args.min_entropy_to_fork,
        "samples_per_fork": args.samples_per_fork,
        "samples_per_expansion": args.samples_per_expansion,
        "num_expansion_rounds": args.num_expansion_rounds,
    }


def get_section_title(method: str) -> str:
    """Get human-readable section title for a method."""
    titles = {
        "simple-sampling": "Simple Sampling",
        "forking-paths": "Forking Paths Algorithm",
        "seeking-entropy": "Entropy-Seeking Algorithm",
    }
    # Fallback: convert method name to title case
    return titles.get(method, method.replace("-", " ").title())


def main() -> None:
    """Parse arguments and run generation with the specified method."""
    args = parse_args()

    config_path = Path(args.config)
    config = GenerationConfig.load(config_path)

    # Apply CLI overrides
    overrides = collect_overrides(args)
    apply_cli_overrides_to_config(config, overrides)

    set_seed(config.seed)

    # Run generation with the selected method
    run_generation_script(
        config=config,
        config_path=config_path,
        method=args.method,
        section_title=get_section_title(args.method),
    )


if __name__ == "__main__":
    main()
