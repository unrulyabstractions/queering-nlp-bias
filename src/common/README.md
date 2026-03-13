# src/common/

Shared utilities, data structures, and infrastructure used across the codebase.

## Directory Structure

```
common/
├── math/               # Mathematical utilities (entropy, diversity, probability)
├── logging/            # Structured logging utilities
├── profiler/           # Performance timing utilities
├── text/               # Text processing (thinking blocks, display)
├── viz/                # Visualization utilities
├── base_schema.py      # Serializable dataclass base with deterministic IDs
├── params_schema.py    # Parameter schemas with CLI-style printing
├── callback_types.py   # Type aliases for callbacks (LogFn, ProgressFn)
├── auto_export.py      # Automatic __init__.py exports
├── token_tree.py       # Tree data structure for token trajectories
├── token_trajectory.py # Individual token sequence representation
├── branching_node.py   # Divergence points in tree
├── binary_fork.py      # Pairwise branch comparison
├── experiment_types.py # Experiment structures (GenerationArm, OutputPaths, etc.)
├── default_config.py   # Default parameter values (single source of truth)
├── device_utils.py     # GPU/CPU/MPS detection and memory tracking
├── file_io.py          # JSON utilities with trailing comma tolerance
├── random_seed.py      # Random seed initialization
├── schema_utils.py     # Safe float conversion (inf/nan handling)
└── viz_utils.py        # Text formatting and statistics (hist, plots, etc.)
```

## Core Abstractions

### BaseSchema (`base_schema.py`)

Foundation for all structured data. Provides:
- **Deterministic ID generation**: `get_id()` returns Blake2b hash of object contents
- **Serialization**: `to_dict()` / `from_dict()` handle nested dataclasses and enums
- **Canonical float rounding**: Floats rounded to 8 decimal places for reproducibility
- **JSON serialization**: `to_string()` for readable output, `from_json()` for loading files

### ParamsSchema (`params_schema.py`)

Extends BaseSchema for parameter objects with CLI-style display:
- Subclasses define `_cli_args: ClassVar[dict[str, str]]` for field → CLI argument mapping
- `print()` displays parameters as CLI arguments
- `get_params_dict()` exports all fields

### Callback Types (`callback_types.py`)

Standardized function signatures:
- `LogFn = Callable[[str], None]` - for logging across pipelines
- `ProgressFn = Callable[[str, int, int], None]` - for progress tracking (name, current, total)

### Auto Export (`auto_export.py`)

Eliminates boilerplate in `__init__.py` files. One-liner setup:

```python
from src.common.auto_export import auto_export
__all__ = auto_export(__file__, __name__, globals())
```

Automatically imports modules, subpackages, and re-exports public names.

## Data Structures

### TokenTree (`token_tree.py`)

Represents multiple token trajectories organized into a tree:
- Detects divergence points (BranchingNode) where trajectories split
- Creates binary forks (BinaryFork) for pairwise group comparison
- Supports dynamic tree building via `from_trajectories()`, `add_trajectory()`, `add_fork_between_groups()`

### TokenTrajectory (`token_trajectory.py`)

A single token sequence with probabilities:
- `token_ids`: list of token IDs
- `logprobs`: log-probability of each token
- `full_logits`: full vocabulary logits (optional, cleared before serialization)
- Text fields: `prefill_text`, `generated_text`, `continuation_text`
- Arm-specific fields: `arm_token_lengths`, `arm_text_lengths`, `arm_idx`

### BranchingNode (`branching_node.py`)

Divergence point where trajectories split:
- `next_token_ids`, `next_token_logprobs`: tokens and probs at divergence
- `branching_token_position`: position in sequence
- `vocab_logits`: full logits for each trajectory at this position (optional)
- `forks_idx`: indices of BinaryForks created from this node

### BinaryFork (`binary_fork.py`)

Pairwise comparison between two branches:
- `next_token_ids`: tuple of (token_a, token_b)
- `next_token_logprobs`: tuple of (logprob_a, logprob_b)
- `arm_idx`: tuple of (group_a, group_b)

## Experiment Types

### GenerationArm (`experiment_types.py`)

An arm configuration for trajectory generation:
- `name`: arm identifier
- `prefill`: text prepended before generation
- `parent_idx`: index of parent arm (for AfterBranch text selection)

### ArmGenerationResult

Result from generating trajectories across all branches:
- `trajectories`: list of GeneratedTrajectory
- `arm_indices`: arm index for each trajectory
- `arm_token_lengths`: token count (prompt + prefill) for each arm
- Properties: `prompt_length`, `trunk_length`, `arm_names`

## Utilities

### device_utils.py
GPU/CPU/MPS detection, memory tracking (CUDA, MPS, RAM), memory logging via `log_memory()`.

### file_io.py
JSON utilities: `save_json()`, `load_json()` (tolerates trailing commas), `ensure_dir()`, path parsing.

### schema_utils.py
Safe float conversion: `safe_float()` handles "Inf", "-Inf", "NaN" strings from JSON.

### random_seed.py
Reproducibility: `set_seed()` initializes random, numpy, and torch.

### default_config.py
Single source of truth for all default parameter values across generation, scoring, and estimation.

### viz_utils.py
Text formatting:
- Text: `truncate()`, `preview()`, `wrap_text()`, `escape_newlines()`
- Floats: `sanitize_float()`, `sanitize_floats()`

## Usage

All public symbols are re-exported at the package level:

```python
from src.common import BaseSchema, TokenTree, TokenTrajectory, ParamsSchema
from src.common import GenerationArm, ArmGenerationResult, OutputPaths
from src.common import device_utils, file_io, random_seed
from src.common.math import perplexity, shannon_entropy
```
