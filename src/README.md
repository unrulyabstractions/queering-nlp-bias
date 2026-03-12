# src/

Core library for trajectory generation, scoring, estimation, and visualization.

## Module Overview

```
src/
├── common/       # Data structures and utilities
├── generation/   # Trajectory generation methods and configuration
├── scoring/      # Structure-based scoring of trajectories
├── estimation/   # Normativity metrics and statistical analysis
├── inference/    # Model backends and language model interfaces
└── viz/          # Visualization of results and experiments
```

## Pipeline Architecture

```
Generation → Scoring → Estimation → Visualization
    │           │          │             │
    │           │          │             └─ viz/
    │           │          ├─ Compute cores, deviance, orientation
    │           │          └─ dynamics/ (drift and horizon analysis)
    │           └─ Structure compliance scoring
    └─ Output: out/<method>/gen_<config>.json
```

Output files follow the pattern:
- `out/<method>/gen_<config>.json` — Generation results
- `out/<method>/score_<config>_<judgment>.json` — Scoring results
- `out/<method>/est_<config>_<judgment>.json` — Estimation results

See methodology docs:
- [GENERATION.md](../GENERATION.md)
- [SCORING.md](../SCORING.md)
- [ESTIMATION.md](../ESTIMATION.md)

## Key Data Structures

| Class | Module | Purpose |
|-------|--------|---------|
| `TokenTree` | common/token_tree.py | Tree of trajectories with branching |
| `TokenTrajectory` | common/token_trajectory.py | Single token sequence with logprobs |
| `GenerationConfig` | generation/generation_config.py | Trajectory generation configuration |
| `ScoringConfig` | scoring/scoring_config.py | Structure scoring configuration |
| `ScoringData` | estimation/estimation_scoring_data.py | Scored trajectories for analysis |
| `ModelRunner` | inference/model_runner.py | Unified interface to language models |

## common/

Data structures and shared utilities.

**Subfolders:**
- `analysis/` — Analysis helper types
- `logging/` — Display and formatting utilities
- `math/` — Mathematical functions (entropy, diversity)
- `profiler/` — Performance measurement utilities
- `text/` — Text processing utilities
- `viz/` — Visualization helpers

**Key files:**
- `base_schema.py` — `BaseSchema` base class for all data structures
- `token_tree.py` — `TokenTree` class for trajectory trees
- `token_trajectory.py` — `TokenTrajectory` class for single sequences
- `experiment_types.py` — `GenerationArm` and experiment configuration types

## generation/

Trajectory generation with multiple methods.

**Methods:**
- `simple-sampling` — Parallel independent sampling
- `forking-paths` — Sequential branching with alternation
- `seeking-entropy` — Entropy-seeking guided sampling
- `just-greedy` — Greedy baseline

**Key files:**
- `generation_config.py` — `GenerationConfig` for defining arm structure and parameters
- `generation_pipeline.py` — `run_generation_pipeline()` entry point
- `generation_output.py` — `GenerationOutput` with serialization
- `methods/` — Method implementations

**Output:** `out/<method>/gen_<config>.json`

## scoring/

Score trajectories against user-defined structures.

**Methods:**
- `categorical` — LLM-based multi-class judgments
- `graded` — LLM-based numerical ratings
- `similarity` — Embedding-based similarity scoring
- `count-occurrences` — Pattern matching

**Key files:**
- `scoring_config.py` — `ScoringConfig` for defining structures and scoring rules
- `scoring_pipeline.py` — `run_scoring_pipeline()` entry point
- `scoring_data.py` — Input/output data structures
- `methods/` — Scoring method implementations

**Output:** `out/<method>/score_<config>_<judgment>.json`

See [README.md](./scoring/README.md) and [EXPLANATION.md](./scoring/EXPLANATION.md).

## estimation/

Estimate normativity metrics from scored trajectories.

**Weighting methods:**
- `prob` — Probability weighting (standard)
- `inv-ppl` — Inverse perplexity weighting
- `uniform` — Uniform baseline

**Subfolders:**
- `methods/` — Weighting method implementations
- `dynamics/` — Drift and horizon analysis
- `logging/` — Display utilities

**Key files:**
- `estimation_pipeline.py` — `run_estimation_pipeline()` entry point
- `estimation_output.py` — `EstimationOutput` with serialization
- `estimation_scoring_data.py` — Load and parse scoring JSON
- `arm_types.py` — Arm classification and ordering

**Output:** `out/<method>/est_<config>_<judgment>.json`

See [README.md](./estimation/README.md) and [EXPLANATION.md](./estimation/EXPLANATION.md).

## inference/

Language model loading and inference.

**Supported backends:**
- HuggingFace — Open-source models (CPU/CUDA)
- MLX — Apple Silicon optimization
- OpenAI — GPT models via API
- Anthropic — Claude models via API

**Key files:**
- `model_runner.py` — `ModelRunner` unified interface
- `embedding_runner.py` — Embedding model support
- `generated_trajectory.py` — Single trajectory from generation

## viz/

Comprehensive visualizations of estimation results.

**Plot types:**
- Core compliance bar charts
- Deviance and diversity trajectories
- Orientation vectors (signed differences)
- Generalized cores (q, r variants)
- Trajectory trees (word/phrase level)
- Cross-method comparisons

**Output:** `out/<method>/` with subdirectories per estimation method

See [README.md](./viz/README.md).

## Design Patterns

- **BaseSchema**: All data classes inherit from `BaseSchema` for serialization
- **Registry Pattern**: Pluggable methods for generation, scoring, estimation
- **Weighting Methods**: Configurable probabilistic weighting in estimation
- **Arm Hierarchy**: Root → Trunk → Branches → Twigs with parental relationships
