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
