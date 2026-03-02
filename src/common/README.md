# src/common/

Core data structures, schemas, and utilities.

## Directory Structure

```
common/
├── analysis/           # Tree analysis and metrics
├── math/               # Entropy, diversity, aggregation functions
├── profiler/           # Performance timing utilities
├── token_tree.py       # Main tree data structure
├── token_trajectory.py # Individual trajectory representation
├── branching_node.py   # Divergence point in tree
├── binary_fork.py      # Pairwise branch comparison
├── base_schema.py      # Serializable dataclass base
└── ...                 # Utilities (log, seed, file_io, etc.)
```

Note: Schemas are in `scripts/schemas/` (not in `common/`).

## Core Data Structures

### TokenTree

The central data structure representing multiple token trajectories organized into a tree.

```python
from src.common.token_tree import TokenTree

# Build from trajectories
tree = TokenTree.from_trajectories(
    trajs=[traj1, traj2, traj3],
    groups_per_traj=[(0,), (0,), (1,)],  # Group membership
    fork_arms=["boy", "girl"],            # Branch labels
    trunk=[0, 1, 2],                      # Shared prefix positions
)

# Decode token IDs to text
tree.decode_texts(runner)

# Clear heavy data for serialization
tree.pop_heavy()
```

Key features:
- Stores trajectories with group membership
- Detects divergence points (branching nodes)
- Creates binary forks for cross-group comparison
- Supports text decoding and memory management

### TokenTrajectory

Individual token sequence with log-probabilities.

```python
from src.common.token_trajectory import TokenTrajectory

traj = TokenTrajectory(
    token_ids=[1, 2, 3, 4],
    logprobs=[0.0, -0.5, -1.2, -0.3],
    logits=[-0.5, -1.2, -0.3],  # Next-token predictions
)

# Properties
len(traj)           # 4
traj.predictions    # [2, 3, 4] (next-token IDs)
traj.prob(pos=2)    # exp(-1.2)
```

### BranchingNode

Where trajectories diverge to different next tokens.

```python
from src.common.branching_node import BranchingNode

node = BranchingNode(
    next_token_ids=[100, 200],
    next_token_logprobs=[-0.5, -1.2],
    position=10,
)
```

### BinaryFork

Pairwise comparison between two branches.

```python
from src.common.binary_fork import BinaryFork

fork = BinaryFork(
    tokens=(100, 200),
    logprobs=(-0.5, -1.2),
    groups=(0, 1),
)
```

## Schemas

Schemas are located in `scripts/schemas/` for use by generation and analysis scripts.

### Generation (`scripts/schemas/generation.py`)

```python
from schemas import GenerationConfig, GenerationOutput

# Load config
config = GenerationConfig.load("trials/generation/test.json")
print(config.model)       # "Qwen/Qwen3-0.6B"
print(config.prompt)      # "Once upon a time..."
print(config.branches)    # ["boy", "girl"]

# Save output
output = GenerationOutput.from_tree(config, model_name, tree, method="sampling")
output.save("out/gen_test.json")
```

### Scoring (`scripts/schemas/scoring.py`)

```python
from schemas import ScoringConfig, JudgmentOutput

# Load scoring config
config = ScoringConfig.load("trials/scoring/animal.json")
print(config.categorical_judgements)  # ["Does this mention an animal?", ...]

# Judgment results
result = JudgmentResult(
    trajectory_id="abc123",
    scores=[1, 0, None],  # Yes, No, Unknown
    raw_responses=["Yes", "No", "..."],
)
```

### Estimation (`scripts/schemas/estimation.py`)

```python
from schemas import EstimationOutput, GroupEstimate

# Group-level analysis
group = GroupEstimate(
    group_idx=0,
    core=[1, 0, 1],           # Majority judgment vector
    orientation=0.85,          # Core agreement strength
    deviance_avg=0.12,         # Mean trajectory deviation
    deviance_var=0.03,         # Deviation variance
)
```

## Analysis

### Tree Analysis (`analysis/`)

Two-pass analysis enriches trees with computed metrics:

```python
from src.common.analysis import analyze_token_tree

# Populates analysis on tree components
analyze_token_tree(tree)

# Access metrics on forks
for fork in tree.forks:
    print(fork.analysis.entropy)
    print(fork.analysis.diversity)
```

### Metrics

| Metric | Location | Description |
|--------|----------|-------------|
| Entropy | math/entropy_diversity/ | Shannon, Rényi entropy |
| Diversity | math/entropy_diversity/ | Effective number of choices |
| Perplexity | math/trajectory_metrics.py | Geometric mean of 1/p |
| Cross-entropy | math/trajectory_metrics.py | Average negative log-prob |

## Base Schema

All schemas inherit from `BaseSchema` for consistent serialization:

```python
from src.common.base_schema import BaseSchema
from dataclasses import dataclass

@dataclass
class MyData(BaseSchema):
    name: str
    value: float

data = MyData(name="test", value=1.5)
data.save("output.json")

loaded = MyData.load("output.json")
```

Features:
- Deterministic ID generation via Blake2b hashing
- JSON serialization/deserialization
- Canonical float rounding for reproducibility
- Nested dataclass support

## Utilities

| File | Description |
|------|-------------|
| `log.py` | Logging with `log()`, `log_section()`, `log_params()` |
| `seed.py` | `set_seed()` for reproducibility |
| `device_utils.py` | GPU/CPU/MPS detection and memory clearing |
| `file_io.py` | JSON loading with comment stripping |
| `schema_utils.py` | Schema validation helpers |
