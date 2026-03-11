# Common Utilities

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This package provides shared infrastructure used throughout the codebase.

## Core Abstractions

### BaseSchema

The foundation for all structured data. Every dataclass that crosses module boundaries or gets serialized should inherit from `BaseSchema`.

**Key features:**
- **Deterministic IDs**: `get_id()` returns a Blake2b hash of the object's contents, ensuring identical objects produce identical IDs regardless of when/where they're created
- **Serialization**: `to_dict()` and `from_dict()` handle nested dataclasses, enums, and special float values (NaN, Inf)
- **Canonical rounding**: Floats are rounded to 8 decimal places using ROUND_HALF_EVEN for reproducibility
- **Type conversion**: `from_dict()` automatically converts nested structures to their proper types

```python
@dataclass
class Experiment(BaseSchema):
    name: str
    params: list[float]
    config: NestedConfig  # Also a BaseSchema

# Deterministic ID for deduplication/caching
exp.get_id()  # "a1b2c3d4..."

# Round-trip serialization
data = exp.to_dict()
restored = Experiment.from_dict(data)
assert exp.get_id() == restored.get_id()
```

### ParamsSchema

Extends BaseSchema for parameter objects, adding CLI-style display:

```python
@dataclass
class TrainingParams(ParamsSchema):
    learning_rate: float
    batch_size: int

    _cli_args: ClassVar[dict[str, str]] = {
        "learning_rate": "--lr",
        "batch_size": "--batch-size",
    }

params.print()
# Output:
#   Parameters:
#     --lr 0.001
#     --batch-size 32
```

### Callback Types

Standardized function signatures for cross-cutting concerns:

```python
LogFn = Callable[[str], None]
# Used for logging throughout pipelines

ProgressFn = Callable[[str, int, int], None]
# (task_name, current, total) for progress bars
```

## Auto-Export System

The `auto_export` function eliminates boilerplate in `__init__.py` files:

```python
# In any package's __init__.py:
from src.common.auto_export import auto_export
__all__ = auto_export(__file__, __name__, globals())
```

This automatically:
1. Imports all `.py` modules in the directory
2. Imports all subpackages (directories with `__init__.py`)
3. Re-exports public names from modules
4. Makes subpackages available as attributes

**What gets exported:**
- All public names (not starting with `_`) from `.py` files
- All subpackages
- Excludes: stdlib modules, third-party libs (numpy, torch, etc.), typing imports

**Import patterns enabled:**
```python
# Flat imports (most common)
from src.common import BaseSchema, TokenTree

# Subpackage imports
from src.common import math
math.perplexity(logprobs)

# Direct module imports (still work)
from src.common.base_schema import BaseSchema
```

## Data Structures

### TokenTree

Represents multiple token trajectories organized into a tree structure:
- Stores trajectories with group membership (e.g., "boy" vs "girl" continuations)
- Detects divergence points (branching nodes) where trajectories split
- Creates binary forks for pairwise comparison between groups

### TokenTrajectory

A single token sequence with log-probabilities:
- `token_ids`: list of token IDs
- `logprobs`: log-probability of each token given context
- Properties: `predictions` (next-token IDs), `prob(pos)` (probability at position)

### BranchingNode

Where trajectories diverge:
- `position`: index in the sequence where divergence occurs
- `next_token_ids`: the different tokens that follow
- `next_token_logprobs`: their log-probabilities

### BinaryFork

Pairwise comparison between two branches:
- `tokens`: (token_a, token_b)
- `logprobs`: (logprob_a, logprob_b)
- `groups`: (group_a, group_b)

## Utility Modules

### log_utils.py

Structured logging with consistent formatting:
- `log_header()`, `log_major()`, `log_stage()` - Section headers
- `log_table_header()` - Formatted table columns
- `log_kv()` - Key-value pairs
- `log_wrapped()` - Word-wrapped text

### file_io.py

JSON loading with comment stripping (allows `//` comments in config files).

### device_utils.py

GPU/CPU/MPS detection and memory management.

### seed.py

Reproducibility via `set_seed()`.

## Design Principles

1. **No nested dicts**: Use `BaseSchema` subclasses instead of `dict[str, dict[...]]`
2. **Auto-export everything**: No manual `__all__` lists in `__init__.py`
3. **Imports at top**: No inline imports except for circular dependency resolution
4. **Unique filenames**: No two `.py` files share a name across the repo
