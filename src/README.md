# src/

Core library for trajectory generation, analysis, and visualization.

## Module Overview

```
src/
├── common/      # Data structures, math, and utilities
├── inference/   # Model backends and trajectory generation
└── viz/         # Tree visualization
```

## Pipeline Architecture

```
Generation → Scoring → Estimation → Visualization
    │           │          │             │
    │           │          │             └─ viz/plot.py
    │           │          └─ common/math/entropy_diversity/
    │           └─ common/token_tree.py + schemas
    └─ inference/model_runner.py
```

See methodology docs for conceptual background:
- [GENERATION.md](../GENERATION.md)
- [SCORING.md](../SCORING.md)
- [ESTIMATION.md](../ESTIMATION.md)

## Key Data Structures

| Class | Module | Description |
|-------|--------|-------------|
| `TokenTree` | common/token_tree.py | Tree of trajectories with branching |
| `TokenTrajectory` | common/token_trajectory.py | Single token sequence with logprobs |
| `BranchingNode` | common/branching_node.py | Divergence point in tree |
| `BinaryFork` | common/binary_fork.py | Pairwise branch comparison |
| `ModelRunner` | inference/model_runner.py | Unified model interface |

## common/

Data structures and mathematical utilities.

### Token Tree

```python
from src.common.token_tree import TokenTree

tree = TokenTree.from_trajectories(
    trajs=[traj1, traj2, traj3],
    groups_per_traj=[(0,), (0,), (1,)],
    fork_arms=["boy", "girl"],
    trunk=[0, 1, 2, ...],
)

# Decode token IDs to text
tree.decode_texts(runner)

# Serialize for output
tree.to_dict()
```

### Math Utilities

```python
from src.common.math.entropy_diversity import (
    generalized_system_core,
    deviance,
    orientation,
    expected_deviance,
)

# Compute probability-weighted core
core = generalized_system_core(compliances, probs, q=1.0, r=1.0)

# Compute deviance from core
dev = deviance(compliance, core, norm="l2")
```

## inference/

Model loading and trajectory generation.

### ModelRunner

```python
from src.inference import ModelRunner

runner = ModelRunner("Qwen/Qwen3-0.6B")

# Properties
runner.device          # "cuda", "mps", or "cpu"
runner.vocab_size      # 151936

# Tokenization
ids = runner.encode_ids("Hello world")
text = runner.decode_ids(ids)

# Generation
traj = runner.generate_trajectory_from_prompt(
    prompt="Write a story",
    prefilling="Once upon a time",
    max_new_tokens=100,
    temperature=1.0,
)
```

### Backends

Automatic backend selection based on model and hardware:

| Backend | Use Case |
|---------|----------|
| HuggingFace | Most open-source models (CUDA/CPU) |
| MLX | Apple Silicon optimization |
| OpenAI | GPT models via API |
| Anthropic | Claude models via API |

## viz/

Tree visualization and plotting.

```python
from src.viz.plot import visualize_experiment

visualize_experiment(
    tree=tree,
    runner=runner,
    mode="word",  # "token", "word", or "phrase"
)
```

## Design Patterns

- **BaseSchema**: All schemas inherit for JSON serialization
- **Backend Abstraction**: Unified interface across model providers
- **Group-Based Analysis**: Trajectories belong to groups for comparison
- **Two-Pass Analysis**: Basic metrics first, then structure-aware analysis
