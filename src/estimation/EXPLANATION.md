# Estimation Algorithm Specification

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This document provides an in-depth explanation of the normativity estimation algorithm, including the mathematical foundations, weighting methods, and data flow.

## Overview

The estimation pipeline computes **normativity metrics** from scored trajectories. The core insight: diversity is always relative to a context (a "system" of structures). What counts as "diverse" depends on which structures we care about.

**Input**: Scored trajectories with structure compliance scores and log probabilities
**Output**: Per-arm statistics including core, deviance, orientation, and their variants

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
| `core` | `<Lambda>` | Primary core (q=1, r=1) |
| `deviance_avg` | `E[d\|B]` | Expected deviance relative to this arm's core |
| `deviance_var` | `Var[d\|B]` | Variance of deviance |
| `deviance_avg_trunk` | `E[d\|T]` | Expected deviance relative to trunk core |
| `deviance_delta` | `E[Delta d]` | `E[d|branch] - E[d|trunk]` |
| `orientation_avg` | `E[theta\|T]` | Expected orientation relative to trunk |
| `orientation_norm` | `\|\|E[theta]\|\|` | L2 norm of expected orientation |
| `core_variants` | - | List of (q, r) core variants with deviances |

### Deviance Delta Interpretation

`deviance_delta = E[d|branch] - E[d|trunk]` measures how a branch changes diversity:
- Positive: Branch increases deviance (more diverse / less normative)
- Negative: Branch decreases deviance (more homogenized / more normative)
- Zero: Branch maintains same deviance level as trunk

### Orientation Norm Interpretation

`orientation_norm = ||E[theta|T]||` measures distance between branch core and trunk core:
- Zero: Branch core equals trunk core
- Large: Branch core is far from trunk core (systematic shift)

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
    +-- branches: list[str]                          # Branch names
    +-- metadata: ScoringMetadata                    # File refs, model info
```

Key methods:
- `group_by_arm()` -> `dict[str, list[TrajectoryScoringData]]`
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
EstimationResult
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
- Saved to JSON: `output.save(path)`
- Summarized to console: `output.summarize()`
- Saved as text summary: `output.save_summary(path)`

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

        # Compute deviance stats
        dev_avg = expected_deviance(structure_scores_list, core, weights)
        dev_var = deviance_variance(structure_scores_list, core, weights)

        # Compute orientation relative to reference (trunk) core
        ref_core = reference_cores.get(method_name) or core
        orient_avg = expected_orientation(structure_scores_list, ref_core, weights)
        orient_norm = ||orient_avg||_2

        # Compute deviance relative to trunk
        dev_avg_trunk = expected_deviance(structure_scores_list, ref_core, weights)
        dev_delta = dev_avg - dev_avg_trunk

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
