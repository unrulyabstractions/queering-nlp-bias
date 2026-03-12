# Adding a New Generation Method

This guide walks you through adding a new generation method to the pipeline.

## Quick Summary

Adding a new method requires **ONE FILE** with:
1. A params dataclass with `name` ClassVar
2. A function decorated with `@register_method`

The method will automatically be available via:
- `run_full_experiment.py --method your-method-name`
- `run_generation_pipeline(runner, config, method="your-method-name")`

## Minimal Example

```python
"""My custom generation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner

from ..generation_config import GenerationConfig
from ..generation_method_registry import GenerationMethodParams, register_method
from src.common.experiment_types import ArmGenerationResult


@dataclass
class MyParams(GenerationMethodParams):
    """Parameters for my method."""

    my_param: int = 10  # All fields must have defaults

    name: ClassVar[str] = "my-method"  # This is the CLI name


@register_method(MyParams)
def generate_my_method(
    runner: ModelRunner,
    config: GenerationConfig,
    params: MyParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate trajectories using my algorithm."""
    from .generation_method_utils import compute_arm_token_lengths

    arms = config.get_arms(runner.skip_thinking_prefix)
    arm_token_lengths = compute_arm_token_lengths(runner, config, arms)

    all_trajectories = []
    all_arm_indices = []

    for arm_idx, arm in enumerate(arms):
        # Your generation logic here
        traj = runner.generate_trajectory_from_prompt(
            prompt=config.prompt,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            prefilling=arm.prefill,
        )
        all_trajectories.append(traj)
        all_arm_indices.append(arm_idx)

    return ArmGenerationResult(
        trajectories=all_trajectories,
        arm_indices=all_arm_indices,
        arms=arms,
        arm_token_lengths=arm_token_lengths,
    )
```

That's it. Save as `src/generation/methods/my_method.py` and it's registered.

## Step-by-Step Guide

### Step 1: Create the Method File

Create a new file in `src/generation/methods/`:

```
src/generation/methods/your_method_name.py
```

### Step 2: Define Parameters

```python
@dataclass
class YourParams(GenerationMethodParams):
    """Parameters for your method."""

    # Your parameters with defaults (required)
    your_param: int = 10
    another_param: float = 0.5

    # REQUIRED: Method name for CLI and registry
    name: ClassVar[str] = "your-method"

    # OPTIONAL: CLI argument mapping for overrides
    _cli_args: ClassVar[dict[str, str]] = {
        "your_param": "--your-param",
        "another_param": "--another-param",
    }
```

### Step 3: Implement and Register

```python
@register_method(YourParams)
def generate_your_method(
    runner: ModelRunner,
    config: GenerationConfig,
    params: YourParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    """Generate trajectories using your algorithm."""
    # Implementation here
    ...
```

## Configuring Method Parameters

Method params can be configured in the JSON config file:

```json
{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Write a story...",
    "method_params": {
        "simple-sampling": {"overrides": {"samples_per_arm": 50}},
        "your-method": {"overrides": {"your_param": 20, "another_param": 0.8}}
    }
}
```

The pipeline accesses params via `config.get_params(method_name)`, which:
1. Creates a default params instance for the method
2. Applies any overrides from `method_params`

## Testing Your Method

```bash
# Check it's registered
uv run python -c "from src.generation import list_methods; print(list_methods())"

# Run with your method
uv run python scripts/run_full_experiment.py \
    --method your-method \
    trials/generation/example.json \
    trials/scoring/example.json
```

## Key Types Reference

### ArmGenerationResult

```python
@dataclass
class ArmGenerationResult:
    trajectories: list[GeneratedTrajectory]  # All generated trajectories
    arm_indices: list[int]                  # Arm index for each trajectory
    arms: list[GenerationArm]               # Arm configuration objects
    arm_token_lengths: list[int]            # Total tokens (prompt + prefill) per arm
```

### GeneratedTrajectory

```python
@dataclass
class GeneratedTrajectory:
    token_ids: list[int]           # Full token sequence
    logprobs: list[float]          # Log-probability per token
    entropies: list[float] | None  # Optional: entropy at each position
```

### GenerationArm

```python
@dataclass
class GenerationArm:
    prefill: str      # Text to prefill (skip_prefix + trunk + optional branch)
    name: str         # "root", "trunk", "branch_N", or "twig_*"
    parent_idx: int   # Index of parent arm
```

## Tips

1. **Use `log_fn` for progress**: Pass messages to `log_fn()` so users can see progress.

2. **Handle empty branches**: `config.branches` may be empty; then you only have trunk.

3. **Respect config settings**: Use `config.temperature` and `config.max_new_tokens`.

4. **If you need separate logging**: Create a `_params.py` file for your params to avoid circular imports between method and logging modules.

5. **Look at existing methods**: `just_greedy_method.py` is the simplest example.
