# src/common/

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Shared utilities, data structures, and mathematical primitives used across the codebase.

## Directory Structure

```
common/
├── math/               # Mathematical utilities (entropy, diversity, probability)
├── analysis/           # Tree analysis and metrics computation
├── logging/            # Structured logging utilities
├── profiler/           # Performance timing utilities
├── text/               # Text processing (display, EOS handling)
├── viz/                # Visualization utilities
├── base_schema.py      # Serializable dataclass base with deterministic IDs
├── params_schema.py    # Parameter schemas with CLI-style printing
├── callback_types.py   # Type aliases for callbacks (LogFn, ProgressFn)
├── auto_export.py      # Automatic __init__.py exports
├── token_tree.py       # Tree data structure for token trajectories
├── token_trajectory.py # Individual token sequence representation
├── branching_node.py   # Divergence points in the tree
├── binary_fork.py      # Pairwise branch comparison
└── ...                 # Other utilities (log, seed, file_io, device_utils)
```

## Key Modules

### BaseSchema (`base_schema.py`)

Base class for all data schemas. Provides:
- Deterministic ID generation via Blake2b hashing
- `to_dict()` / `from_dict()` serialization
- Canonical float rounding for reproducibility
- Nested dataclass support

```python
from src.common import BaseSchema

@dataclass
class MyData(BaseSchema):
    name: str
    value: float

data = MyData(name="test", value=1.5)
data.get_id()  # Deterministic hash
data.to_dict() # Serializable dict
```

### ParamsSchema (`params_schema.py`)

Extension of BaseSchema for parameter objects with CLI-style printing:

```python
from src.common import ParamsSchema

@dataclass
class MyParams(ParamsSchema):
    count: int
    _cli_args = {"count": "--count"}

params.print()  # Displays as CLI arguments
```

### Callback Types (`callback_types.py`)

Standard callback signatures:
- `LogFn = Callable[[str], None]` - logging callback
- `ProgressFn = Callable[[str, int, int], None]` - progress callback (name, current, total)

### Auto Export (`auto_export.py`)

Automatic re-exporting of public names from submodules:

```python
# In any __init__.py
from src.common.auto_export import auto_export
__all__ = auto_export(__file__, __name__, globals())
```

## Usage

All public symbols are re-exported at the package level:

```python
from src.common import BaseSchema, TokenTree, TokenTrajectory
from src.common.math import q_diversity, shannon_entropy, perplexity
```

See `math/EXPLANATION.md` for mathematical utilities documentation.
