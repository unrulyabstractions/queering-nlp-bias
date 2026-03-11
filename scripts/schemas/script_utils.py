"""Shared utilities for pipeline scripts.

This module provides common functions used across generation, scoring,
estimation, and experiment scripts to reduce code duplication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.log import log, log_params, log_section
from src.common.log_utils import (
    HEADER_WIDTH,
)
from src.common.seed import set_seed
from src.generation import (
    GenerationConfig,
    OutputPaths,
)
from src.generation.generation_config import MethodParamsOverride
from src.generation.generation_method_registry import (
    get_params_class,
    list_methods,
)
from src.inference import ModelRunner


def apply_cli_overrides_to_config(
    config: GenerationConfig,
    overrides: dict[str, Any],
) -> None:
    """Apply CLI argument overrides to config's method_params.

    Maps CLI argument names to method params using each method's _cli_args.
    This keeps CLI handling in the script layer, not in GenerationConfig.

    Args:
        config: The generation config to modify
        overrides: Dict of CLI arg names (snake_case) to values
    """
    for method_name in list_methods():
        params_class = get_params_class(method_name)
        cli_args = getattr(params_class, "_cli_args", {})

        # Check each CLI arg mapping
        for field_name, cli_name in cli_args.items():
            # Convert --foo-bar to foo_bar for lookup
            override_key = cli_name.lstrip("-").replace("-", "_")
            if overrides.get(override_key) is not None:
                # Ensure method has an entry in method_params
                if method_name not in config.method_params:
                    config.method_params[method_name] = MethodParamsOverride()
                config.method_params[method_name].overrides[field_name] = overrides[
                    override_key
                ]


def log_output_paths(paths: OutputPaths, gap: int = 0) -> None:
    """Log output file paths."""
    log("Output files:", gap=gap)
    log(f"  -> {paths.generation}")
    log(f"  -> {paths.judgment}")
    log(f"  -> {paths.estimation}")


def log_experiment_start(
    title: str,
    paths: OutputPaths,
    **params: Any,
) -> None:
    """Log experiment header with params and output paths."""
    log("═" * HEADER_WIDTH)
    log(title)
    log("═" * HEADER_WIDTH)
    if params:
        log_params(**params)
    log_output_paths(paths, gap=1)
    log("═" * HEADER_WIDTH)


# ══════════════════════════════════════════════════════════════════════════════
# Argument Parsing
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ArgSpec:
    """Specification for an extra command-line argument."""

    name: str
    type: type
    metavar: str
    help: str


@dataclass
class ParsedArgs:
    """Result of parsing generation script arguments."""

    config: GenerationConfig
    config_path: Path


def parse_generation_args(
    description: str,
    examples: list[str],
    extra_args: list[ArgSpec] | None = None,
) -> ParsedArgs:
    """Parse command-line arguments for generation scripts.

    Args:
        description: Script description
        examples: List of example usage lines
        extra_args: Additional arguments specific to this script

    Returns:
        ParsedArgs with config (CLI overrides already applied) and config_path
    """
    epilog = "Examples:\n" + "\n".join(f"  %(prog)s {ex}" for ex in examples)

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog,
    )
    parser.add_argument("config", help="Path to generation config JSON")

    if extra_args:
        for arg in extra_args:
            parser.add_argument(
                f"--{arg.name}",
                type=arg.type,
                metavar=arg.metavar,
                help=arg.help,
            )

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    config = GenerationConfig.load(config_path)

    # Apply CLI overrides to config
    if extra_args:
        overrides = {
            arg.name.replace("-", "_"): getattr(args, arg.name.replace("-", "_"))
            for arg in extra_args
        }
        apply_cli_overrides_to_config(config, overrides)

    set_seed(config.seed)

    return ParsedArgs(config=config, config_path=config_path)


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════


def load_model(config: GenerationConfig) -> ModelRunner:
    """Load and validate the model from config."""
    if not config.model:
        raise ValueError("No model specified in generation config")

    log(f"Loading model: {config.model}")
    runner = ModelRunner(config.model)

    # Get model type from runner (it detects chat models)
    model_type = "CHAT/INSTRUCT" if runner._is_chat_model else "BASE"

    box_content = f"MODEL TYPE: {model_type}"
    log(f"\n  ╔{'═' * (len(box_content) + 4)}╗")
    log(f"  ║  {box_content}  ║")
    log(f"  ╚{'═' * (len(box_content) + 4)}╝\n")
    return runner


def log_prompt_header(prompt: str, trunk: str, branches: list[str]) -> None:
    """Log the prompt and structure at the start of generation.

    Args:
        prompt: The user prompt
        trunk: The trunk/shared prefix text
        branches: List of branch continuation texts (e.g., [" boy", " cat"])
    """
    log_section("Generation Setup")
    log("  Prompt:")
    for line in prompt.split("\n"):
        log(f"    {line}")
    log(f'\n  Trunk (shared prefix): "{trunk}"')
    if branches:
        log(f"  Branches ({len(branches)}): {branches}")
        log(f"  Total groups: {len(branches) + 1} (trunk + {len(branches)} branches)")
