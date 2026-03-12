# Estimation Algorithm Specification

This document provides an in-depth explanation of the normativity estimation algorithm, including the mathematical foundations, weighting methods, and data flow.

## Overview

The estimation pipeline computes **normativity metrics** from scored trajectories. The core insight: diversity is always relative to a context (a "system" of structures). What counts as "diverse" depends on which structures we care about.

**Input**: Scored trajectories with structure compliance scores and log probabilities
**Output**: Per-arm statistics including core, deviance, and core variants

## Core Concepts

### Structure and Compliance

A **structure** is a specification of organization among tokens (e.g., "mentions women", "uses formal language"). Each trajectory receives a **structure compliance** score:

```
alpha_i(x) in [0, 1]  -- How much string x satisfies structure i
```

The **system compliance** is the vector of all structure compliances:

```
Lambda_n(x) = [alpha_1(x), alpha_2(x), ..., alpha_n(x)]
```

### System Core

The **system core** is the expected system compliance under a distribution:

```
<Lambda_n> = E[Lambda_n(x)] = sum_x p(x) * Lambda_n(x)
```

This represents the "normal" - what the distribution expects for each structure.

### Orientation and Deviance

**Orientation** measures how a trajectory deviates from the core:

```
theta_n(x) = Lambda_n(x) - <Lambda_n>
```

Positive components indicate over-compliance, negative indicates under-compliance.

**Deviance** is the scalar magnitude of non-normativity:

```
d_n(x) = ||theta_n(x)||_2 = sqrt(sum_i (alpha_i(x) - <alpha_i>)^2)
```

Higher deviance = more "queer" / non-normative relative to the distribution.

## Generalized (q, r) Core Parameterization

The standard core uses arithmetic mean with actual probability weights. The generalized formulation allows different views:

```
<Lambda_n>_{q,r} = M_q(Lambda_n(x), escort_r(p(x)))
```

Where:
- **r** controls which trajectories get attention (escort distribution)
- **q** controls how compliance values are aggregated (power mean)

### Escort Distribution (r parameter)

The escort distribution reweights probabilities:

```
escort_r(p) = p^r / sum_x p(x)^r
```

Special cases:
- `r = 1`: Actual distribution (standard)
- `r = 0`: Uniform over support
- `r = inf`: Mode (most probable)
- `r = -inf`: Anti-mode (rarest)

### Power Mean (q parameter)

The weighted power mean aggregates compliance values:

```
M_q(values, weights) = (sum_i w_i * v_i^q)^(1/q)
```

Special cases:
- `q = 1`: Arithmetic mean (standard)
- `q = 0`: Geometric mean (sensitive to exclusion)
- `q = -1`: Harmonic mean (penalizes low compliance)
- `q = inf`: Maximum
- `q = -inf`: Minimum

### Named Core Variants

The system computes multiple named (q, r) combinations:

| Name | q | r | Interpretation |
|------|---|---|----------------|
| standard | 1 | 1 | Expected compliance under actual distribution |
| uniform | 1 | 0 | Average over support (ignores probability) |
| mode | 1 | inf | Compliance of the most probable trajectory |
| max | inf | 1 | Maximum compliance in distribution |
| mode_min | -inf | inf | Minimum compliance among modes |
| confident | 1 | 2 | Upweights high-probability trajectories |
| rms | 2 | 1 | Root-mean-square (emphasizes high compliance) |
| antimode | 1 | -inf | Compliance of rarest trajectory |
| geometric | 0 | 1 | Geometric mean (penalizes zero compliance) |
| harmonic | -1 | 1 | Harmonic mean (penalizes low compliance) |

## Weighting Methods

Weighting methods convert `(log_probs, n_tokens)` into normalized weights for computing cores and deviances.

### Probability Weighting (`prob`)

**File**: `methods/prob_weighting_method.py`

Normalizes log probabilities to proper probabilities:

```
w_i = exp(log_p_i) / sum_j exp(log_p_j)
```

This is the standard approach - trajectories contribute proportionally to their probability under the model.

### Inverse Perplexity Weighting (`inv-ppl`)

**File**: `methods/inv_ppl_weighting_method.py`

Weights by model confidence per token:

```
inv_ppl_i = exp(log_p_i / n_tokens_i)
w_i = inv_ppl_i / sum_j inv_ppl_j
```

This normalizes by sequence length:
- Long sequences with low per-token probability get lower weight
- Short confident sequences get higher weight

Useful when sequence length varies significantly and you want to focus on per-token model confidence rather than total probability.

### Uniform Weighting (`uniform`)

**File**: `methods/uniform_weighting_method.py`

Equal weight for all trajectories:

```
w_i = 1 / n
```

Ignores probability entirely. Serves as a baseline for comparison.

## Registry Pattern

**File**: `weighting_method_registry.py`

New weighting methods are registered via decorator with an ENABLED flag:

```python
# Set to False to disable this weighting method
ENABLED = True

@dataclass
class MyWeightingParams(WeightingMethodParams):
    name: ClassVar[str] = "my-method"
    description: ClassVar[str] = "my-method-weighted"

def compute_my_weights(log_probs, n_tokens, params) -> list[float]:
    # Must return normalized weights summing to 1.0
    ...

if ENABLED:
    compute_my_weights = register_method(MyWeightingParams)(compute_my_weights)
```

To disable a method, set `ENABLED = False` at the top of its file.

Registry functions:
- `get_method(name)` - Get weight function by name
- `get_default_params(name)` - Get default params instance
- `list_methods()` - List all registered method names
- `iter_methods()` - Iterate (name, params_class, fn) tuples

Methods are auto-discovered when the `methods/` package is imported.

## Computed Metrics

For each arm and weighting method, the pipeline computes:

### WeightedEstimate Fields

| Field | Symbol | Description |
|-------|--------|-------------|
| `core` | `Λ` | Primary core (q=1, r=1) |
| `deviance_avg` | `E[∂\|self]` | Expected deviance relative to this arm's core |
| `deviance_var` | `Var[∂\|self]` | Variance of deviance |
| `deviance_avg_root` | `E[∂\|root]` | Expected deviance relative to root core |
| `deviance_avg_trunk` | `E[∂\|trunk]` | Expected deviance relative to trunk core |
| `orientation_from_root` | `θ(arm\|root)` | Orientation vector relative to root core |
| `orientation_norm_from_root` | `\|\|θ\|\|` | Magnitude of orientation from root |
| `orientation_from_trunk` | `θ(arm\|trunk)` | Orientation vector relative to trunk core |
| `orientation_norm_from_trunk` | `\|\|θ\|\|` | Magnitude of orientation from trunk |
| `orientation_from_parent` | `θ(arm\|parent)` | Orientation relative to parent branch (twigs only) |
| `orientation_norm_from_parent` | `\|\|θ\|\|` | Magnitude of orientation from parent branch |
| `core_variants` | - | List of (q, r) core variants with deviances |

### Derived Metrics (Computed on Demand)

These values are computed from cores when needed, not stored:
- **Deviance delta**: `E[∂|self] - E[∂|trunk]` (change vs trunk)

## Data Flow

### Input: ScoringData

**File**: `estimation_scoring_data.py`

```
scoring JSON file
    |
    v
ScoringData.load(path)
    |
    +-- scoring_data: dict[str, list[ScoringItem]]  # config_key -> items
    +-- results: list[dict]                          # Per-trajectory scores
    +-- arm_names: list[str]                         # Arm names (root, trunk, branch_1, ...)
    +-- metadata: ScoringMetadata                    # File refs, model info
```

Key methods:
- `group_by_arm()` -> `dict[str, list[TrajectoryScoringData]]`
- `get_all_trajectories()` -> `list[TrajectoryScoringData]` (all trajectories across arms)
- `get_structure_scores(result)` -> `list[float]`  (compliance vector)
- `get_structure_info()` -> `list[StructureInfo]`

### Processing: Pipeline

**File**: `estimation_pipeline.py`

```
ScoringData
    |
    v
run_estimation_pipeline(data, judgment_file)
    |
    +-- Group trajectories by arm
    |
    +-- Compute trunk estimate (reference cores)
    |       |
    |       +-- For each weighting method:
    |               compute_weighted_estimate()
    |
    +-- Compute branch estimates (relative to trunk cores)
    |       |
    |       +-- For each weighting method:
    |               compute_weighted_estimate(reference_core=trunk_core)
    |
    v
PipelineResult
    +-- output: EstimationOutput
    +-- arms: list[ArmEstimate]
    +-- trunk_cores: dict[str, list[float]]
```

### Output: ArmEstimate

**File**: `estimation_structure.py`

```
ArmEstimate
    +-- arm_idx: int
    +-- name: str
    +-- trajectories: list[TrajectoryEstimate]
    +-- estimates: dict[str, WeightedEstimate]  # method_name -> estimate
```

Each `WeightedEstimate` contains all metrics for one weighting method.

### Serialization: EstimationOutput

**File**: `estimation_output.py`

The output can be:
- Saved to JSON: `output.save(path)` (goes to `out/<method>/<gen_name>/<scoring_name>/` subfolder)
- Summarized to console: `output.summarize()`
- Saved as text summary: `output.save_summary(path)` (goes to `out/<method>/<gen_name>/<scoring_name>/` subfolder)

Example output paths:
```
out/simple-sampling/example/example/estimation.json
out/simple-sampling/example/example/summary_estimation.txt
out/simple-sampling/example/example/viz/           # Visualizations
out/simple-sampling/example/example/viz/dynamics/  # Dynamics plots
```

### Trajectory Text Field

Each `TrajectoryScoringData` now includes a `text` field containing the continuation text. This is used for:
- Dynamics analysis (drift and horizon computation)
- Output storage for downstream processing
- Visualization and reporting

## Dynamics Analysis: Drift, Horizon, and Pull

**Files**: `dynamics/dynamics_types.py`, `dynamics/dynamics_computation.py`, `dynamics/dynamics_visualization.py`

The dynamics module analyzes how trajectories evolve relative to reference cores. Three metrics capture complementary aspects:

**Drift y(k)**: Deviance of PARTIAL text (re-scored) relative to root core
- Re-scores partial text at token position k
- Shows how far a trajectory has deviated from root as text develops
- Plotted as a continuous curve over token position (purple)

**Horizon z(arm)**: Deviance of FULL trajectory relative to each arm's core along trajectory's path
- Uses pre-computed full trajectory scores (no re-scoring)
- Only computes for arms on the trajectory's ancestry path (see below)
- Plotted at each arm's prefix token count (blue, connected line)

**Pull x(arm)**: L2 norm of arm's core at each arm's prefix position
- Represents the "strength" of normative characterization at each arm
- Plotted alongside horizon for comparison (orange, connected line)

### Arm Ancestry

A trajectory only computes horizon and pull for arms on its **ancestry path**. This is determined by `get_arm_ancestry()` from `arm_types.py`:

| Trajectory Arm | Ancestry Path |
|----------------|---------------|
| `root` | `["root"]` |
| `trunk` | `["root", "trunk"]` |
| `branch_1` | `["root", "trunk", "branch_1"]` |
| `branch_2` | `["root", "trunk", "branch_2"]` |
| `twig_2_b1` | `["root", "trunk", "branch_1", "twig_2_b1"]` |

### Output Structure

Dynamics plots are saved to:
```
out/<method>/<gen_name>/<scoring_name>/viz/dynamics/traj_{idx}_{arm}.png
```

Each plot shows three curves:
- Drift (purple): deviance from root as text develops
- Horizon (blue): deviance from each arm's core at arm prefix positions
- Pull (orange): L2 norm of arm's core at arm prefix positions

### Example Usage

```python
from src.estimation.dynamics import compute_dynamics, plot_dynamics

# Given an EstimationResult and ScoringConfig
dynamics_result = compute_dynamics(
    estimation_result=result,
    scoring_config=scoring_config,
    trajs_per_arm=1,  # Analyze 1 trajectory per arm
)

for traj_dyn in dynamics_result.trajectories:
    print(f"Trajectory {traj_dyn.traj_idx} ({traj_dyn.arm_name}): {traj_dyn.n_tokens} tokens")

    # Drift points (re-scored partial text vs root)
    for drift_point in traj_dyn.drift_points:
        print(f"  Drift @{drift_point.token_position}: deviance={drift_point.deviance:.4f}")

    # Horizon points (full text vs each arm's core on ancestry path)
    for hp in traj_dyn.horizon_points:
        marker = "*" if hp.arm_name == traj_dyn.arm_name else ""
        print(f"  Horizon({hp.arm_name}, @{hp.arm_prefix_tokens}): {hp.deviance:.4f}{marker}")

    # Pull points (L2 norm of each arm's core)
    for pp in traj_dyn.pull_points:
        print(f"  Pull({pp.arm_name}, @{pp.arm_prefix_tokens}): ||core||={pp.pull:.4f}")

# Generate visualization
output_dir = Path("out/simple-sampling/gen_name/score_name/viz/dynamics")
saved_paths = plot_dynamics(dynamics_result, output_dir)
```

For complete details, see [dynamics/EXPLANATION.md](./dynamics/EXPLANATION.md).

## Algorithm: compute_arm_estimate()

```python
def compute_arm_estimate(arm_idx, name, trajectories, reference_cores=None):
    # Extract data
    structure_scores_list = [t.structure_scores for t in trajectories]
    log_probs = [t.conditional_logprobs.get(name, 0.0) for t in trajectories]
    n_tokens = [t.n_continuation_tokens for t in trajectories]

    estimates = {}
    for method_name in iter_methods():
        # Get weights from weighting function
        weight_fn = get_method(method_name)
        weights = weight_fn(log_probs, n_tokens, params)

        # Compute core (q=1, r=1)
        core = generalized_system_core(structure_scores_list, weights, q=1, r=1)

        # Compute deviance stats relative to own core
        dev_avg = expected_deviance(structure_scores_list, core, weights)
        dev_var = deviance_variance(structure_scores_list, core, weights)

        # Compute deviance relative to reference cores
        dev_avg_root = expected_deviance(structure_scores_list, root_core, weights)
        dev_avg_trunk = expected_deviance(structure_scores_list, trunk_core, weights)

        # Pre-compute orientation vectors relative to reference cores
        orientation_from_root = core - root_core  # vector subtraction
        orientation_from_trunk = core - trunk_core
        orientation_from_parent = core - parent_core  # for twigs only

        # Compute (q, r) variants
        core_variants = compute_core_variants(structure_scores_list, weights)

        estimates[method_name] = WeightedEstimate(...)

    return ArmEstimate(arm_idx, name, trajectories, estimates)
```

## Mathematical Reference

### Expected Deviance

```
E[d] = sum_i w_i * ||Lambda_n(x_i) - <Lambda_n>||_2
```

Low E[d] indicates homogenization - trajectories cluster near the core.
High E[d] indicates diversity - trajectories spread far from the core.

### Deviance Variance

```
Var[d] = E[d^2] - E[d]^2
```

High variance means some trajectories are very normative while others are very deviant.

### Orientation Computation

```
E[theta] = sum_i w_i * (Lambda_n(x_i) - <Lambda_n>_ref)
```

When computed relative to trunk core, this shows the systematic direction of shift.

### Core Variants

For each (q, r) pair:

```
<Lambda_n>_{q,r} = [
    M_q([alpha_1(x_i)], escort_r([p_i])),
    M_q([alpha_2(x_i)], escort_r([p_i])),
    ...
]
```

Each variant provides a different view of "what is normal" under the distribution.
