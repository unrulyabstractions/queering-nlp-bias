# CLAUDE.md - src/generation/

This module generates token trajectories using various sampling strategies.

## Architecture

```
GenerationConfig --> run_generation_pipeline() --> GenerationPipelineResult
                            |
                            v
                    get_method() --> generate_fn()
                            |
                            v
                    ArmGenerationResult --> TokenTree --> GenerationOutput
```

## Registry Pattern

**Adding a new generation method requires ONE FILE in `methods/`.**

```python
# methods/my_sampling_method.py
from dataclasses import dataclass
from typing import ClassVar
from src.common.experiment_types import ArmGenerationResult
from ..generation_method_registry import GenerationMethodParams, register_method

@dataclass
class MyParams(GenerationMethodParams):
    name: ClassVar[str] = "my-method"
    samples_per_arm: int = 10

@register_method(MyParams)
def generate_my_method(runner, config, params, log_fn=None):
    trajectories = []
    arm_indices = []

    for arm in config.get_arms():
        for _ in range(params.samples_per_arm):
            traj = runner.generate_trajectory_from_prompt(
                config.prompt + arm.prefill,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
            )
            trajectories.append(traj)
            arm_indices.append(arm.arm_idx)

    return ArmGenerationResult(
        trajectories=trajectories,
        arm_indices=arm_indices,
        trunk_length=config.get_trunk_length(runner),
        prompt_length=config.get_prompt_length(runner),
    )
```

## Key Files

| File | Purpose |
|------|---------|
| `generation_pipeline.py` | `run_generation_pipeline()` entry point |
| `generation_method_registry.py` | `@register_method`, `get_method()`, `list_methods()` |
| `generation_config.py` | `GenerationConfig` with prompt, arms, method params |
| `generation_output.py` | `GenerationOutput` serialization |
| `generation_helpers.py` | Output formatting utilities |

## Available Methods

| Method | Description | Key Params |
|--------|-------------|------------|
| `simple-sampling` | N samples per arm | `samples_per_arm` |
| `forking-paths` | Greedy + entropy-based exploration | `min_entropy`, `min_prob`, `max_alternates` |
| `seeking-entropy` | Iterative expansion at high-entropy points | `num_expansion_rounds`, `samples_per_expansion` |
| `just-greedy` | One greedy path per arm | (none) |

## Arms and Forks

Arms define conditioning prefixes:
- `trunk`: shared prefix for all branches
- `branch_N`: branch-specific continuations
- `twig_M_bN`: optional finer distinctions (twig M of branch N)

Fork arms define which branch pairs to compare (for BinaryFork creation).

```python
config.get_arms()  # Returns list[GenerationArm]
config.fork_arms   # Returns list[tuple[int, int]]
```

## Config Structure

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "User prompt...",
  "trunk": "Shared continuation",
  "branches": ["Option A", "Option B"],
  "temperature": 1.0,
  "max_new_tokens": 128,
  "method_params": {
    "simple-sampling": {"overrides": {"samples_per_arm": 20}}
  }
}
```

## Method Logging

Methods in `methods/logging/` provide structured output:
- `forking_paths_logging.py` - fork analysis, candidate display

Use the `log_fn` callback for consistent formatting.

## Common Pitfalls

1. **Return ArmGenerationResult** - not raw trajectories
2. **Use config.get_arms()** - includes trunk and all branches
3. **Calculate trunk_length correctly** - use `config.get_trunk_length(runner)`
4. **All params need defaults** - for `get_default_params()` to work

## Output Path Convention

```
out/<method>/<trial-name>/generation.json
out/<method>/<trial-name>/generation_cfg.json
out/<method>/<trial-name>/summary_generation.txt
```

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - algorithm specifications
- [ADDING_METHOD.md](./ADDING_METHOD.md) - step-by-step guide
- [methods/README.md](./methods/README.md) - method implementations
- [Root CLAUDE.md](../../CLAUDE.md) - global project rules

## Workflow Orchestration

1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
