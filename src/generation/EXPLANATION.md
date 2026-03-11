# Generation Package: In-Depth Specification

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This document provides detailed explanations of the generation algorithms, data flow, key data structures, and the registry pattern.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Data Flow](#data-flow)
3. [Key Data Structures](#key-data-structures)
4. [Registry Pattern](#registry-pattern)
5. [Generation Methods](#generation-methods)
   - [Simple Sampling](#simple-sampling)
   - [Forking Paths](#forking-paths)
   - [Entropy Seeking](#entropy-seeking)
   - [Just Greedy](#just-greedy)
6. [Configuration Options](#configuration-options)

---

## Architecture Overview

The generation package follows a registry-based plugin architecture. Methods self-register via decorator, and the pipeline dispatches to them by name.

```
                                    +-----------------------+
                                    |   GenerationConfig    |
                                    |  (prompt, arms, etc.) |
                                    +-----------+-----------+
                                                |
                                                v
+------------------+    method name    +--------+---------+
|  Method Registry |<------------------| run_generation   |
|  _REGISTRY dict  |                   | _pipeline()      |
+--------+---------+                   +--------+---------+
         |                                      |
         | get_method(name)                     |
         v                                      v
+--------+---------+                   +--------+---------+
| generate_fn()    |------------------>| ArmGeneration    |
| (method impl)    |                   | Result           |
+------------------+                   +--------+---------+
                                                |
                                                v
                                       +--------+---------+
                                       | TokenTree        |
                                       | (tree structure) |
                                       +--------+---------+
                                                |
                                                v
                                       +--------+---------+
                                       | GenerationOutput |
                                       | (serializable)   |
                                       +------------------+
```

---

## Data Flow

### Pipeline Entry Point

```
run_generation_pipeline(runner, config, method="simple-sampling")
         |
         +---> get_method(method) -> generate_fn
         |
         +---> config.get_params(method) -> params instance
         |
         +---> generate_fn(runner, config, params, log_fn)
         |            |
         |            +---> ArmGenerationResult
         |                       |
         +---> _build_token_tree(result, config, runner)
         |            |
         |            +---> TokenTree
         |
         +---> GenerationOutput.from_tree(...)
         |
         +---> GenerationPipelineResult(result, tree, output)
```

### Arm Processing Flow

Each generation method processes arms (trunk + branches) as follows:

```
                    GenerationConfig
                          |
                          v
                   config.get_arms()
                          |
          +---------------+---------------+
          |               |               |
          v               v               v
       Arm[0]          Arm[1]          Arm[N]
      (trunk)        (branch_1)      (branch_N)
          |               |               |
          +---------------+---------------+
                          |
                          v
              Method-specific generation
                          |
                          v
               ArmGenerationResult
         (trajectories + arm_indices)
```

### Token Tree Construction

```
ArmGenerationResult.trajectories
         |
         v
TokenTree.from_trajectories(
    trajs=trajectories,
    groups_per_traj=[(idx,) for idx in arm_indices],
    fork_arms=config.fork_arms,
    trunk=range(trunk_length),
    prompt_length=prompt_length
)
         |
         v
TokenTree with:
  - trajs: list of trajectory dicts
  - nodes: tree structure for visualization
  - fork_arms: pairs for comparison
```

---

## Key Data Structures

### GenerationConfig

Configuration for trajectory generation experiments.

```python
@dataclass
class GenerationConfig(BaseSchema):
    prompt: str              # User prompt text
    model: str               # Model identifier (e.g., "Qwen/Qwen3-0.6B")
    trunk: str               # Shared prefix for all branches
    branches: list[str]      # Branch-specific continuations
    temperature: float       # Sampling temperature (default: 1.0)
    max_new_tokens: int      # Max tokens to generate (default: 128)
    seed: int | None         # Random seed
    method_params: dict[str, MethodParamsOverride]  # Per-method overrides
```

Relationships:
```
GenerationConfig
       |
       +---> get_arms() -> list[Arm]
       |         |
       |         +---> Arm(prefill, name, arm_index)
       |
       +---> fork_arms -> list[tuple[int, int]]  # branch index pairs
       |
       +---> get_params(method) -> GenerationMethodParams
                  |
                  +---> Applies overrides from method_params
```

### GenerationArm

```python
@dataclass
class GenerationArm:
    prefill: str    # Full prefill: skip_prefix + trunk + branch
    name: str       # "trunk" or "branch_N"
    arm_index: int  # 0 for trunk, N for branch_N
```

Fork arms are represented as `tuple[int, int]` (left branch index, right branch index).

### ArmGenerationResult

Output from any generation method:

```python
@dataclass
class ArmGenerationResult:
    trajectories: list[GeneratedTrajectory]  # All generated trajectories
    arm_indices: list[int]                  # Arm index per trajectory
    trunk_length: int                         # Tokens in trunk portion
    prompt_length: int                        # Tokens in prompt only
```

### GenerationMethodParams

Base class for method-specific parameters:

```python
@dataclass
class GenerationMethodParams(ParamsSchema):
    name: ClassVar[str]  # Method name (e.g., "simple-sampling")
```

All subclasses must define `name` as a ClassVar and provide defaults for all fields.

---

## Registry Pattern

The registry pattern enables automatic method discovery without explicit imports.

### How It Works

```
                    @register_method(ParamsClass)
                              |
                              v
                    +------------------+
                    | _REGISTRY dict   |
                    | {                |
                    |   "method-name": |
                    |     (ParamsClass,|
                    |      generate_fn)|
                    | }                |
                    +------------------+
                              |
                              v
              +---------------+---------------+
              |               |               |
              v               v               v
       get_method()    get_params_class() get_default_params()
```

### Registration Mechanism

```python
def register_method(params_class: type[GenerationMethodParams]):
    def decorator(fn_or_cls):
        # Store (params_class, generate_fn) pair in _REGISTRY
        _REGISTRY[params_class.name] = (params_class, fn_or_cls)
        return fn_or_cls
    return decorator
```

### Registry Functions

| Function | Description |
|----------|-------------|
| `get_method(name)` | Returns the generate function |
| `get_default_params(name)` | Returns params instance with all defaults |
| `get_params_class(name)` | Returns the params class type |
| `list_methods()` | Returns sorted list of method names |
| `params_from_dict(method, data)` | Creates params from dict |

### Auto-Discovery

Methods are discovered via Python's module system. When `src.generation.methods` is imported, the `__init__.py` auto-exports all submodules, which triggers their `@register_method` decorators.

```python
# In src/generation/methods/__init__.py
# Auto-export pattern loads all *_method.py files
# Each file's @register_method decorator fires on import
```

---

## Generation Methods

### Simple Sampling

**Method name:** `simple-sampling`

**Algorithm:**
```
For each arm in [trunk, branch_1, ..., branch_N]:
    1. Construct formatted prompt with arm prefill
    2. For i in range(samples_per_arm):
         a. Generate trajectory using temperature sampling
         b. Store trajectory with arm's arm_index
    3. Return all trajectories
```

**Data Flow:**
```
                    config.get_arms()
                          |
        +-----------------+-----------------+
        v                 v                 v
     Arm[0]            Arm[1]            Arm[N]
        |                 |                 |
        v                 v                 v
sample_from_arm()  sample_from_arm()  sample_from_arm()
        |                 |                 |
        +--------+--------+--------+--------+
                          |
                          v
               ArmGenerationResult
        (N_arms * samples_per_arm trajectories)
```

**Parameters:**
```python
@dataclass
class SamplingParams(GenerationMethodParams):
    samples_per_arm: int = 10  # Trajectories per arm
    name: ClassVar[str] = "simple-sampling"
```

**Key Operations:**
- `runner.generate_trajectory_from_prompt()` - generates single trajectory
- Temperature sampling introduces stochasticity

---

### Forking Paths

**Method name:** `forking-paths`

**Algorithm:**
```
For each arm:
    Step 1: Generate greedy path (temperature=0)
            |
            v
    [t0]-[t1]-[t2]-[t3]-[t4]-[t5]...  (greedy tokens)

    Step 2: Analyze all positions (single forward pass)
            - Compute entropy at each position
            - Get top-K candidates with probabilities
            |
            v
    Position 0: entropy=1.2, candidates=[t0(0.8), a(0.1), b(0.05)]
    Position 1: entropy=2.5, candidates=[t1(0.4), c(0.3), d(0.2)]
    ...

    Step 3: Find qualifying forks
            - Filter: entropy >= min_entropy
            - Filter: alternate prob >= min_prob
            - Limit: max_alternates per position
            |
            v
    Qualifying: [(pos=1, alt=c), (pos=1, alt=d), (pos=3, alt=x)]

    Step 4: Expand each fork point
            - Build prefix: prompt + greedy[0:pos] + alternate
            - Sample N continuations from that prefix
            |
            v
    [t0]-[c]-[...continuation 1...]
             [...continuation 2...]
    [t0]-[d]-[...continuation 1...]
             [...continuation 2...]
```

**Visual Representation:**
```
               [greedy path]
    [t0]--[t1]--[t2]--[t3]--[t4]--[t5]--[EOS]
           |
           +--[c]--[...cont1...]
           |       [cont2]
           +--[d]--[...cont3...]
                   [cont4]
```

**Data Structures:**
```python
@dataclass
class TopKCandidate:
    token_id: int
    prob: float
    logprob: float

@dataclass
class PositionAnalysis:
    position: int
    entropy: float
    greedy_token_id: int
    candidates: list[TopKCandidate]

@dataclass
class QualifyingFork:
    analysis: PositionAnalysis
    candidate: TopKCandidate

@dataclass
class ForkPoint:
    position: int
    entropy: float
    greedy_token_id: int
    alternate: TopKCandidate
    continuations: list[GeneratedTrajectory]
```

**Parameters:**
```python
@dataclass
class ForkingParams(GenerationMethodParams):
    max_alternates: int = 3     # Max alternate tokens per position
    min_prob: float = 0.1       # Min probability for alternate
    min_entropy: float = 0.5    # Min entropy to consider forking
    samples_per_fork: int = 2   # Continuations per fork point
    name: ClassVar[str] = "forking-paths"
```

**Key Operations:**
- `analyze_all_positions()` - single forward pass for entropy + top-K
- `find_qualifying_forks()` - filter by entropy and probability thresholds
- `expand_fork_point()` - sample continuations from alternate token

---

### Entropy Seeking

**Method name:** `seeking-entropy`

**Algorithm:**
```
For each arm:
    Step 1: Initialize tree with N sampled trajectories
            - Sample N trajectories
            - Compute entropy at all positions via forward pass
            |
            v
    Path 0: [t0,t1,t2,...] entropies=[1.2, 2.5, 0.8, ...]
    Path 1: [t0,t1,t3,...] entropies=[1.2, 2.5, 1.1, ...]
    ...

    Step 2: Expansion loop (K rounds)
        For round in 1..K:
            a. Find (path, position) with highest unused entropy
            b. Mark position as used on that path
            c. Build prefix: path.tokens[0:position+1]
            d. Sample N new trajectories from prefix
            e. Compute entropies for new trajectories
            f. Add new paths to tree
            |
            v
    After K rounds: tree has N + K*N trajectories

    Step 3: Return all trajectories from tree
```

**Visual Representation:**
```
Round 0 (initialization):
    Path0: [t0]--[t1]--[t2]--[t3]--[EOS]
                  ^
                  |entropy=2.5 (highest)
    Path1: [t0]--[t1]--[t4]--[t5]--[EOS]

Round 1 (expand at Path0, pos=1):
    Path0: [t0]--[t1]--[t2]--[t3]--[EOS]
                  |
    Path2:        +--[a]--[b]--[c]--[EOS]
    Path3:        +--[d]--[e]--[f]--[EOS]
                     ^
                     |entropy=3.1 (now highest)

Round 2 (expand at Path3, pos=2):
    ...
```

**Data Structures:**
```python
@dataclass
class TreePath:
    trajectory: GeneratedTrajectory
    path_id: int
    entropies: list[float]
    continuation: str
    parent_id: int | None
    branch_pos: int | None
    used_positions: set[int]

    def best_unused_position(self, prompt_len) -> BestPosition:
        """Find highest-entropy unused position."""

@dataclass
class ExpansionPoint:
    path: TreePath | None
    position: int | None
    entropy: float
```

**Parameters:**
```python
@dataclass
class EntropySeekingParams(GenerationMethodParams):
    samples_per_expansion: int = 3  # Trajectories per expansion
    num_expansion_rounds: int = 5   # Number of expansion rounds
    name: ClassVar[str] = "seeking-entropy"
```

**Key Operations:**
- `compute_entropies()` - single forward pass for all position entropies
- `find_best_expansion_point()` - scan all paths for highest unused entropy
- `initialize_tree()` - sample initial trajectories with entropies
- `expand_tree()` - iterative expansion loop

---

### Just Greedy

**Method name:** `just-greedy`

**Algorithm:**
```
For each arm:
    1. Generate one trajectory with temperature=0
    2. Return single trajectory with arm's arm_index
```

This is the simplest method, useful for baseline comparisons.

**Parameters:**
```python
@dataclass
class JustGreedyParams(GenerationMethodParams):
    # No configurable parameters
    name: ClassVar[str] = "just-greedy"
```

**Output:** One trajectory per arm (N_arms total).

---

## Configuration Options

### GenerationConfig JSON

```json
{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Write a story about a cat.",
    "trunk": "Once upon a time",
    "branches": [" there was a brave cat", " there was a lazy cat"],
    "temperature": 1.0,
    "max_new_tokens": 128,
    "seed": 42,
    "method_params": {
        "simple-sampling": {
            "overrides": {"samples_per_arm": 20}
        },
        "forking-paths": {
            "overrides": {
                "max_alternates": 5,
                "min_prob": 0.05,
                "min_entropy": 1.0,
                "samples_per_fork": 3
            }
        },
        "seeking-entropy": {
            "overrides": {
                "samples_per_expansion": 5,
                "num_expansion_rounds": 10
            }
        }
    }
}
```

### Parameter Resolution

```
config.get_params("forking-paths")
         |
         v
get_default_params("forking-paths")
         |
         v
ForkingParams()  # defaults from class
         |
         v
method_params["forking-paths"].apply_to(params)
         |
         v
ForkingParams with overrides applied
```

### CLI Arguments

Each params class can define `_cli_args` for command-line override mapping:

```python
_cli_args: ClassVar[dict[str, str]] = {
    "samples_per_arm": "--samples-per-arm",
    "max_alternates": "--max-alternates-per-position",
}
```

This enables:
```bash
python scripts/run_full_experiment.py \
    --method forking-paths \
    --max-alternates-per-position 5 \
    config.json
```
