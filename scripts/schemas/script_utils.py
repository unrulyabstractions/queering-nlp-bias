"""Shared utilities for pipeline scripts.

This module provides common functions used across generation, scoring,
estimation, and experiment scripts to reduce code duplication.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.common.logging import HEADER_WIDTH, log, log_params, log_section
from src.common.random_seed import set_seed
from src.common.experiment_types import OutputPaths
from src.generation import (
    GenerationConfig,
    GenerationOutput,
    run_generation_pipeline,
)
from src.generation.generation_config import MethodParamsOverride
from src.generation.generation_method_registry import (
    get_params_class,
    list_methods,
)
from src.generation.methods.logging import log_tree_trajectories
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


def log_prompt_header(
    prompt: str,
    trunk: str,
    branches: list[str],
    twig_variations: list[str] | None = None,
) -> None:
    """Log the prompt and structure at the start of generation.

    Args:
        prompt: The user prompt
        trunk: The trunk text
        branches: List of branch continuation texts (e.g., [" boy", " cat"])
        twig_variations: Optional list of twig variation texts
    """
    log_section("Generation Setup")
    log("  Prompt:")
    for line in prompt.split("\n"):
        log(f"    {line}")
    log(f'\n  Trunk: "{trunk}"')
    if branches:
        log(f"  Branches ({len(branches)}): {branches}")
    if twig_variations:
        log(f"  Twig variations ({len(twig_variations)}): {twig_variations}")

    # Count arms: root + trunk + branches + (branches * twig_variations)
    n_branches = len(branches)
    n_twigs = n_branches * len(twig_variations) if twig_variations else 0
    total_arms = 2 + n_branches + n_twigs  # root + trunk + branches + twigs

    parts = ["root", "trunk"]
    if n_branches:
        parts.append(f"{n_branches} branches")
    if n_twigs:
        parts.append(f"{n_twigs} twigs")
    log(f"  Total arms: {total_arms} ({', '.join(parts)})")


# ══════════════════════════════════════════════════════════════════════════════
# Generation Pipeline Runner
# ══════════════════════════════════════════════════════════════════════════════


def run_generation_script(
    config: GenerationConfig,
    config_path: Path,
    method: str,
    section_title: str,
) -> GenerationOutput:
    """Run a complete generation script pipeline.

    This shared function handles the common pattern used by all generate_by_*.py
    scripts:
        1. Load model
        2. Log section header and parameters
        3. Log prompt structure
        4. Run generation pipeline
        5. Log trajectory details
        6. Save output and summary files
        7. Display generation summary

    Args:
        config: The generation configuration
        config_path: Path to the config file (used for output paths)
        method: The generation method name (e.g., "simple-sampling")
        section_title: Title for the log section (e.g., "Simple Sampling")

    Returns:
        The GenerationOutput from the pipeline
    """
    runner = load_model(config)

    log_section(section_title)
    config.get_params(method).print()

    log_prompt_header(config.prompt, config.trunk, config.branches, config.twig_variations)

    # Run generation pipeline with logging
    result = run_generation_pipeline(runner, config, method=method, log_fn=log)

    # Log trajectory details
    log_tree_trajectories(result.result, runner)

    # Save outputs (and copy original config)
    n_trajs = len(result.result.trajectories)
    output_path = GenerationOutput.compute_output_path(config_path, method=method)
    result.output.save(output_path, config_path=config_path)
    log(f"\nSaved {n_trajs} trajectories to {output_path}")

    summary_path = GenerationOutput.compute_summary_path(config_path, method=method)
    result.output.save_summary(summary_path)
    log(f"Saved summary to {summary_path}")

    # Show summary
    result.output.summarize()

    return result.output
