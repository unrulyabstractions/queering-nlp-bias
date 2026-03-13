# Common Utilities

This package provides shared infrastructure used throughout the codebase.

## Core Abstractions

### BaseSchema

The foundation for all structured data. Every dataclass that crosses module boundaries or gets serialized should inherit from `BaseSchema`.

**Key features:**
- **Deterministic IDs**: `get_id()` returns a Blake2b hash of the object's contents, ensuring identical objects produce identical IDs
- **Serialization**: `to_dict()` and `from_dict()` handle nested dataclasses, enums, and special float values (NaN, Inf)
- **Canonical rounding**: Floats are rounded to 8 decimal places using ROUND_HALF_EVEN for reproducibility
- **JSON support**: `to_string()` for readable output, `from_json()` for loading from files
- **Hook system**: `_to_dict_hook()` allows subclasses to customize serialization (e.g., prob expansion in BranchingNode)

```python
@dataclass
class Experiment(BaseSchema):
    name: str
    params: list[float]

# Deterministic ID for deduplication
exp.get_id()  # "a1b2c3d4..."

# Round-trip serialization
data = exp.to_dict()
restored = Experiment.from_dict(data)
assert exp.get_id() == restored.get_id()
```

### ParamsSchema

Extends BaseSchema for parameter objects, adding CLI-style display. Subclasses define `_cli_args` to map field names to CLI argument names:

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

Standardized function signatures for logging and progress:

```python
LogFn = Callable[[str], None]
# Used for logging throughout pipelines

ProgressFn = Callable[[str, int, int], None]
# (task_name, current, total) for progress tracking
```

## Auto-Export System

The `auto_export` function eliminates boilerplate in `__init__.py` files. One-liner setup:

```python
# In any package's __init__.py:
from src.common.auto_export import auto_export
__all__ = auto_export(__file__, __name__, globals())
```

**What it does:**
1. Imports all `.py` modules in the directory
2. Imports all subpackages (directories with `__init__.py`)
3. Re-exports public names (not starting with `_`) from modules
4. Makes subpackages available as attributes

**What gets excluded from exports:**
- Stdlib modules (sys, os, json, etc.)
- Third-party packages (numpy, torch, pandas, etc.)
- Typing constructs (Any, Callable, etc.)
- Dataclass helpers (field, asdict, etc.)

**Import patterns enabled:**
```python
# Flat imports (most common)
from src.common import BaseSchema, TokenTree, ParamsSchema

# Subpackage imports
from src.common import math
math.perplexity(logprobs)

# Direct module imports (still work)
from src.common.base_schema import BaseSchema
```

## Data Structures

### TokenTree

Represents multiple token trajectories organized into a tree:
- **Trajectories**: `trajs` - list of TokenTrajectory objects with group membership
- **Nodes**: `nodes` - BranchingNode objects at divergence points
- **Forks**: `forks` - BinaryFork objects for pairwise group comparisons
- **Metadata**: `trunk_length`, `prompt_length`, `trunk_text`, `fork_arms`

**Key methods:**
- `from_trajectories()` - build tree from trajectories with group assignments
- `add_trajectory()` - add a trajectory, recalculate nodes/forks
- `add_fork_between_groups()` - add a fork relationship
- `get_logits_at_node()` - retrieve logits from first trajectory at node
- `pop_heavy()` - clear full_logits and vocab_logits for serialization
- `decode_texts()` - decode trunk and continuation text

### TokenTrajectory

A single token sequence with log-probabilities and metadata:

**Core fields:**
- `token_ids`: list of token IDs in the sequence
- `logprobs`: log-probability of each token
- `logits`: logit (max logit) for each token
- `full_logits`: full vocabulary logits (torch.Tensor, cleared before serialization)

**Text fields:**
- `prefill_text`: trunk/branch/twig text prepended before generation
- `generated_text`: text the model generated
- Properties: `continuation_text` (prefill + generated), `continuation_text_no_thinking` (strips `<think>...</think>`)

**Metadata:**
- `arm_token_lengths`: token count (prompt + prefill) for each arm
- `arm_text_lengths`: character count for each arm's prefill
- `arm_idx`: which groups this trajectory belongs to
- `nodes_idx`: indices of BranchingNodes this trajectory passes through
- `traj_idx`: index in parent tree

**Text extraction:**
- `text_after_arm(arm_idx)` - get continuation text after a specific arm's prefill
- `get_conditional_prob(start, end)` - probability of substring

### BranchingNode

A divergence point where trajectories split:

**Core fields:**
- `next_token_ids`: tuple of token IDs chosen at this divergence
- `next_token_logprobs`: tuple of log-probabilities for each token
- `branching_token_position`: position in sequence where divergence occurs

**Data:**
- `traj_idx`: indices of trajectories passing through this node
- `vocab_logits`: full logits from each trajectory at this position (optional)
- `forks_idx`: indices of BinaryForks created from this node
- `node_idx`: index in parent tree

**Serialization:**
- Implements `_to_dict_hook()` to summarize `vocab_logits` as `"[N items]"` instead of full arrays

### BinaryFork

Pairwise comparison between two branches:

**Core fields:**
- `next_token_ids`: tuple of (token_a, token_b)
- `next_token_logprobs`: tuple of (logprob_a, logprob_b)
- `arm_idx`: tuple of (group_a, group_b) for the two branches
- `fork_idx`: index in parent tree

## Experiment Types

### GenerationArm

Configuration for one arm of the generation tree:
- `name`: arm identifier (e.g., "prompt", "trunk", "branch_boy", "branch_girl")
- `prefill`: text to prepend before generation
- `parent_idx`: index of parent arm (for AfterBranch text selection)

### ArmGenerationResult

Result from generating trajectories across all branches:
- `trajectories`: list of GeneratedTrajectory objects
- `arm_indices`: arm index for each trajectory
- `arm_token_lengths`: token count (prompt + prefill) for each arm
- `arms`: full GenerationArm objects with prefills

**Properties:**
- `prompt_length`: length of just the prompt (root arm)
- `trunk_length`: length of prompt + trunk
- `arm_names`: arm names in index order

### OutputPaths

Container for output paths throughout the experiment pipeline:
- `generation`: generation output directory
- `judgment`: scoring/judgment output directory
- `estimation`: estimation results directory

## Utility Modules

### device_utils.py

GPU/CPU/MPS detection and memory management:
- `get_device()` - return best available device (cuda, mps, or cpu)
- `get_memory_usage()` - dict with cuda/mps/ram memory stats
- `log_memory()` - print memory usage at a stage
- `clear_gpu_memory()` - empty GPU caches

### file_io.py

JSON utilities with robustness to formatting:
- `save_json()` - pretty-print JSON (converts multiline text to arrays)
- `load_json()` - load JSON (restores multiline text, tolerates trailing commas and double commas)
- `parse_file_path()` - flexible path parsing (simple name, filename, or full path)
- `ensure_dir()` - create directory if needed
- `get_timestamp()` - current timestamp string

### schema_utils.py

Safe float conversion for JSON round-trip:
- `safe_float()` - convert value to float, handling "Inf", "-Inf", "NaN" strings

### random_seed.py

Reproducibility:
- `set_seed(seed)` - initialize random, numpy, and torch seeds

### default_config.py

Single source of truth for all default parameter values:
- **Generation**: TEMPERATURE, MAX_NEW_TOKENS, SAMPLING_SAMPLES_PER_ARM, FORKING_* params, ENTROPY_* params
- **Scoring**: JUDGE_MAX_TOKENS, STRING_SELECTION
- **Embedding**: EMBEDDING_MODEL
- **Estimation**: DEFAULT_STATISTIC, DEFAULT_WEIGHTING_METHOD
- **Dynamics**: DYNAMICS_STEP, DYNAMICS_TRAJS_PER_ARM, DYNAMICS_ARMS

### viz_utils.py

Text formatting and visualization for console output:

**Text utilities:**
- `escape_newlines()` - convert newlines to `\\n`
- `truncate()` - truncate to max_len with suffix
- `preview()` - escape newlines and truncate
- `wrap_text()` - word-wrap to width

**Float utilities:**
- `sanitize_float()` - replace inf/nan with finite values
- `sanitize_floats()` - apply to list

## Design Principles

1. **All dataclasses inherit from BaseSchema**: Ensures serialization, deterministic IDs, and round-trip safety
2. **Auto-export everything**: No manual `__all__` lists; auto_export handles it
3. **All imports at top**: No inline imports except circular dependency resolution
4. **No nested dicts**: Use BaseSchema subclasses instead of `dict[str, dict[...]]`
5. **Unique filenames**: No two `.py` files share a name across the repo
