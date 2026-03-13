# CLAUDE.md - src/common/

This module provides shared infrastructure used across the entire codebase. **Always check here first** before creating new utilities.

## Core Patterns

### BaseSchema (base_schema.py)
**Every dataclass that holds structured data MUST inherit from BaseSchema.**

```python
from src.common import BaseSchema

@dataclass
class MyResult(BaseSchema):
    score: float
    labels: list[str]
```

Provides:
- `get_id()` - deterministic Blake2b hash for deduplication
- `to_dict()` / `from_dict()` - automatic serialization with nested dataclass support
- `_to_dict_hook()` - override for custom serialization (e.g., prob expansion)

### ParamsSchema (params_schema.py)
For parameter objects with CLI-style printing. Extends BaseSchema.

```python
from src.common import ParamsSchema

@dataclass
class MyParams(ParamsSchema):
    threshold: float = 0.5
    _cli_args: ClassVar[dict[str, str]] = {"threshold": "--threshold"}
```

### Auto-Export (__init__.py pattern)
All `__init__.py` files use auto-export. When adding new files, **no changes to __init__.py are needed**.

```python
from src.common.auto_export import auto_export
__all__ = auto_export(__file__, __name__, globals())
```

## Key Data Structures

| File | Class | Purpose |
|------|-------|---------|
| `token_tree.py` | TokenTree | Tree of token trajectories with divergence points |
| `token_trajectory.py` | TokenTrajectory | Single token sequence with logprobs |
| `branching_node.py` | BranchingNode | Divergence point where trajectories split |
| `binary_fork.py` | BinaryFork | Pairwise comparison between two branches |
| `experiment_types.py` | GenerationArm, ArmGenerationResult, OutputPaths | Experiment structures |

## Subpackages

| Subpackage | Purpose | Key Exports |
|------------|---------|-------------|
| `math/` | Entropy, diversity, probability utilities | `perplexity()`, `shannon_entropy()`, `diversity_index()` |
| `logging/` | Structured console output | `log()`, `log_header()`, `log_table_row()` |
| `profiler/` | Performance timing | `@timed`, `ProfileTimer` |
| `text/` | Text processing | EOS handling, thinking block extraction |
| `viz/` | Text-based visualization | `tree_display.py` for ASCII trees |

## Utility Files

| File | When to use |
|------|-------------|
| `file_io.py` | JSON with trailing/double comma tolerance, path parsing |
| `device_utils.py` | GPU/MPS/CPU detection, memory tracking |
| `random_seed.py` | Setting seeds (random, numpy, torch) |
| `default_config.py` | **Single source of truth** for all default values (generation, scoring, estimation, dynamics) |
| `callback_types.py` | `LogFn`, `ProgressFn` type aliases |

## Common Pitfalls

1. **Do NOT create new functions without searching first** - EXHAUSTIVELY search this module before creating ANY new utility function. Use grep/glob to find existing implementations:
   - `math/vector_utils.py` - L2 norm, L2 distance, orientation vectors
   - `math/entropy_diversity/` - deviance, entropy, divergence, diversity
   - `math/probability_utils.py` - log prob normalization, perplexity
   - `text/` - string processing, EOS handling, thinking blocks
   - `file_io.py` - JSON loading, path utilities
   - `device_utils.py` - memory tracking, GPU/MPS operations
2. **Do NOT create new utils files** - check if functionality exists in `math/`, `text/`, or `file_io.py`
3. **Do NOT use nested dicts** - use BaseSchema subclasses instead
4. **Do NOT inline import** - all imports go at top of file
5. **Do NOT forget BaseSchema** - every crossing-boundary dataclass needs it

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - comprehensive API reference
- [README.md](./README.md) - directory structure overview
- [Root CLAUDE.md](../../CLAUDE.md) - global project rules
