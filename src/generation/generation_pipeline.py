"""Core generation pipeline logic.

This module contains the complete generation pipeline that can be used
programmatically without any logging dependencies.

The pipeline uses the method registry to dispatch to the appropriate
generation method implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.common.callback_types import LogFn, ProgressFn
from src.common.experiment_types import ArmGenerationResult
from src.common.profiler import profile
from src.common.token_tree import TokenTree
from src.inference import ModelRunner

from .generation_config import GenerationConfig
from .generation_method_registry import get_method, get_output_name
from .generation_output import GenerationOutput


@dataclass
class GenerationPipelineResult:
    """Result of running the generation pipeline."""

    result: ArmGenerationResult
    tree: TokenTree
    output: GenerationOutput


@profile
def run_generation_pipeline(
    runner: ModelRunner,
    config: GenerationConfig,
    method: str = "sampling",
    progress_fn: ProgressFn | None = None,
    log_fn: LogFn | None = None,
) -> GenerationPipelineResult:
    """Run the generation pipeline.

    Single entry point for all generation methods. Uses the method registry
    to dispatch to the appropriate implementation.

    Args:
        runner: Model runner for generation
        config: Generation config with prompt, arms, and parameters
        method: Generation method name (any registered method)
        progress_fn: Optional callback(name, current, total) for progress
        log_fn: Optional logging callback

    Returns:
        GenerationPipelineResult with result, tree, and output
    """
    # Get method function and params
    generate_fn = get_method(method)
    params = config.get_params(method)

    # Run generation via method function
    result = generate_fn(runner, config, params, log_fn)

    # Set arm_idx on each trajectory and build tree
    for traj, arm_idx_val in zip(result.trajectories, result.arm_indices):
        traj.arm_idx = (arm_idx_val,)

    # Build token tree from trajectories
    tree = TokenTree.from_trajectories(
        trajs=result.trajectories,
        trunk=list(range(result.trunk_length)),
        prompt_length=result.prompt_length,
    )
    tree.decode_texts(runner)

    # Create output structure (output_name from registry)
    output = GenerationOutput.from_tree(
        config=config,
        model=runner.model_name,
        tree=tree,
        arms=result.arms,
        method=get_output_name(method),
        eos_token=runner.eos_token,
        arm_token_lengths=result.arm_token_lengths,
    )

    return GenerationPipelineResult(result=result, tree=tree, output=output)
