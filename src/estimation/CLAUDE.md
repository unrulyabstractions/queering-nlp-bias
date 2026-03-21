# CLAUDE.md - src/estimation/

This module computes normativity metrics (core, deviance, orientation) from scored trajectories.

## Core Concepts

- **Structure compliance**: How much a trajectory satisfies each scoring structure (0-1)
- **System core**: Expected compliance vector under a distribution
- **Deviance**: Distance from core (non-normativity measure)
- **Orientation**: Direction of deviation from reference arm's core

## Architecture

```
ScoringData --> run_estimation_pipeline() --> EstimationResult
                       |
                       v
              compute_arm_estimate() --> WeightedEstimate (per weighting method)
                       |
                       v
              ArmEstimate (per arm, all methods)
```

## Registry Pattern (Weighting Methods)

**Adding a new weighting method requires ONE FILE in `methods/`.**

```python
# methods/my_weighting_method.py
from dataclasses import dataclass
from typing import ClassVar
from ..weighting_method_registry import WeightingMethodParams, register_method

ENABLED = True  # Set to False to disable

@dataclass
class MyWeightingParams(WeightingMethodParams):
    name: ClassVar[str] = "my-weighting"
    description: ClassVar[str] = "my-weighted"

def compute_my_weights(log_probs, n_tokens, params):
    # Return normalized weights summing to 1.0
    return [1.0 / len(log_probs)] * len(log_probs)

if ENABLED:
    compute_my_weights = register_method(MyWeightingParams)(compute_my_weights)
```

## Key Files

| File | Purpose |
|------|---------|
| `estimation_pipeline.py` | `run_estimation_pipeline()`, `compute_arm_estimate()` |
| `weighting_method_registry.py` | `@register_method`, `get_method()`, `iter_methods()` |
| `arm_types.py` | `ArmKind`, `classify_arm()`, `get_arm_color()`, `get_arm_ancestry()` |
| `estimation_scoring_data.py` | `ScoringData` - input loading |
| `estimation_weighted_types.py` | `WeightedEstimate` - results per method |
| `estimation_structure.py` | `ArmEstimate`, `TrajectoryScoringData` |
| `estimation_core_types.py` | `CoreVariant`, `NAMED_CORES` (q,r) params |

## Available Weighting Methods

| Method | Description |
|--------|-------------|
| `prob` | Normalize by probability (default) |
| `inv-ppl` | Inverse perplexity (per-token confidence) |
| `uniform` | Equal weight (baseline) |

## Arm Types and Ordering

Arms have a canonical order: root -> trunk -> branches -> twigs

```python
from src.estimation.arm_types import classify_arm, get_arm_color, get_arm_ancestry

classify_arm("branch_1")  # ArmKind.BRANCH
get_arm_color("trunk")    # "#E67E22"
get_arm_ancestry("twig_2_b1")  # ["root", "trunk", "branch_1", "twig_2_b1"]
```

## WeightedEstimate Fields

| Field | Symbol | Meaning |
|-------|--------|---------|
| `core` | Lambda | Expected structure compliance |
| `deviance_avg` | E[d\|self] | Average deviance from own core |
| `deviance_avg_root` | E[d\|root] | Average deviance from root core |
| `orientation_from_trunk` | theta(arm\|trunk) | Direction of shift from trunk |

## Dynamics Module

The dynamics module (at `src/dynamics/`) analyzes how trajectories evolve:
- **Drift**: Deviance of partial text from root core
- **Potential**: Deviance of full text from each arm's core
- **Pull**: L2 norm of arm's core

See [../dynamics/EXPLANATION.md](../dynamics/EXPLANATION.md).

## Common Pitfalls

1. **Weighting functions must return normalized weights** - sum to 1.0
2. **Use `ENABLED` flag** - not deletion, to disable methods
3. **Arm ancestry matters** - potential/pull only computed for arms on trajectory's path
4. **Reference cores come from trunk** - orientation is relative to trunk

## Output Format (v2.0)

EstimationOutput is the official, versioned output format:

```python
EstimationOutput:
    metadata: EstimationMetadata  # version, timestamp, source files, models
    structures: list[StructureInfo]  # structure definitions
    arms: list[ArmEstimate]  # per-arm cores and deviance
    arm_scoring: list[ArmScoring]  # per-arm compliance rates
```

## Output Path Convention

```
out/<gen-method>/<gen-name>/<scoring-name>/estimation.json
out/<gen-method>/<gen-name>/<scoring-name>/summary_estimation.txt
out/<gen-method>/<gen-name>/<scoring-name>/viz/dynamics/
```

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - mathematical specification
- [ADDING_METHOD.md](./ADDING_METHOD.md) - adding weighting methods
- [../dynamics/EXPLANATION.md](../dynamics/EXPLANATION.md) - dynamics algorithm
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
