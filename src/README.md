# src/

Core library for trajectory generation, analysis, and visualization.

## Module Overview

```
src/
├── common/      # Core data structures, schemas, and utilities
├── inference/   # Model backends and trajectory generation
└── viz/         # Tree visualization and plotting
```

## Architecture

The system follows a pipeline architecture:

```
Generation → Judgment → Estimation → Visualization
```

1. **Generation** (`inference/`): Load model, sample trajectories, build token tree
2. **Judgment** (`common/schemas/`): Score trajectories on categorical dimensions
3. **Estimation** (`common/schemas/`): Compute normativity metrics from judgments
4. **Visualization** (`viz/`): Render trees with probability annotations

## Key Data Structures

| Class | Module | Description |
|-------|--------|-------------|
| `TokenTree` | common/token_tree.py | Tree of token sequences with branching points |
| `TokenTrajectory` | common/token_trajectory.py | Single token sequence with logprobs |
| `ModelRunner` | inference/model_runner.py | Unified interface for model inference |
| `GenerationConfig` | common/schemas/ | Generation experiment configuration |
| `ScoringConfig` | common/schemas/ | Judgment experiment configuration |

## Design Patterns

- **Dataclass Inheritance**: All schemas inherit from `BaseSchema` for JSON serialization
- **Backend Abstraction**: Model inference abstracted to support HuggingFace and MLX
- **Group-Based Analysis**: Trajectories belong to groups (branches) for comparison
- **Two-Pass Analysis**: Basic metrics first, then structure-aware analysis

## See Also

- [common/README.md](common/README.md) - Data structures and schemas
- [inference/README.md](inference/README.md) - Model backends
- [viz/README.md](viz/README.md) - Visualization tools
