# Estimation Package

Estimate normativity metrics from scored trajectories.

## Directory Structure

```
estimation/
├── methods/                          # Weighting method implementations
│   ├── prob_weighting_method.py         # Probability weighting (default)
│   ├── inv_ppl_weighting_method.py      # Inverse perplexity weighting
│   └── uniform_weighting_method.py      # Uniform weighting (baseline)
├── logging/                          # Display and logging utilities
│   ├── estimation_display_utils.py      # Structure/compliance/core logging
│   ├── estimation_step_logging.py       # Pipeline step logging
│   └── estimation_comparison_logging.py # Cross-method comparison display
├── dynamics/                         # Drift, horizon, and pull analysis
│   ├── logging/                         # Dynamics logging utilities
│   │   └── dynamics_step_logging.py       # Step-by-step drift/horizon/pull logging
│   ├── dynamics_types.py                # DriftPoint, HorizonPoint, PullPoint types
│   ├── dynamics_computation.py          # Compute drift/horizon/pull metrics
│   ├── dynamics_visualization.py        # Plotting dynamics curves
│   ├── README.md                        # Dynamics module overview
│   └── EXPLANATION.md                   # Dynamics algorithm specification
├── arm_types.py                      # Arm classification, colors, ordering utilities
├── weighting_method_registry.py      # Registry pattern for weighting methods
├── estimation_pipeline.py            # Main estimation pipeline (compute_arm_estimate)
├── estimation_output.py              # Output structure and serialization
├── estimation_scoring_data.py        # Input data loading (ScoringData)
├── estimation_structure.py           # ArmEstimate, TrajectoryScoringData types
├── estimation_weighted_types.py      # WeightedEstimate type
├── estimation_core_types.py          # CoreVariant, NAMED_CORES (q,r) params
├── estimation_auxiliary_types.py     # Summary and continuation helper types
├── estimation_scoring_result.py      # StructureInfo, ArmScoring types
└── estimation_experiment_types.py    # EstimationResult for comparisons
```

## Arm Types

The pipeline supports multiple arm types with a defined ordering:

| Arm | Description | Index (if root present) |
|-----|-------------|------------------------|
| `root` | Prompt-only conditioning (no trunk text) | 0 |
| `trunk` | Trunk-only conditioning (reference for orientation) | 1 |
| `branch_N` | Branch conditioning (diverges from trunk) | 2+ |
| `twig_bN_M` or `twig_M_bN` | Twig variation under branch `N` with twig index `M` | n/a |

Use `arm_types.py` for centralized arm handling:

```python
from src.estimation.arm_types import (
    classify_arm,       # Returns ArmKind enum
    is_baseline_arm,    # True for root or trunk
    is_reference_arm,   # True for trunk only
    get_arm_color,      # Get hex color for arm
    sort_arm_names,     # Sort arms in canonical order
)
```

## Quick Start

```python
from src.estimation import ScoringData, run_estimation_pipeline

# Load scoring data (judgment file from scoring pipeline)
scoring_data = ScoringData.load("out/simple-sampling/example/example/scoring.json")

# Run estimation pipeline
result = run_estimation_pipeline(scoring_data, "out/simple-sampling/example/example/scoring.json")

# Access results - iterate over arms and weighting methods
for arm in result.arms:
    for method_name, estimate in arm.estimates.items():
        print(f"{arm.name} [{method_name}]: core={estimate.core}, E[d]={estimate.deviance_avg}")

# Save results (output goes to out/<method>/<gen_name>/<scoring_name>/ subfolder)
result.output.save(result.output.compute_output_path(Path("out/simple-sampling/example/example/scoring.json")))
```

## Key Files

| File | Purpose |
|------|---------|
| `estimation_pipeline.py` | Core algorithm: `run_estimation_pipeline()`, `compute_arm_estimate()` |
| `arm_types.py` | Arm classification, colors, ordering: `ArmKind`, `classify_arm()`, `get_arm_color()` |
| `weighting_method_registry.py` | Registry for pluggable weighting methods |
| `estimation_scoring_data.py` | Load and process scoring JSON files |
| `estimation_weighted_types.py` | `WeightedEstimate` - results for one weighting method (core, deviance, orientation) |
| `estimation_structure.py` | `ArmEstimate` - aggregate across all weighting methods |

## Adding a New Weighting Method

Create one file in `methods/`:

```python
# Set to False to disable this weighting method
ENABLED = True

@dataclass
class MyWeightingParams(WeightingMethodParams):
    name: ClassVar[str] = "my-weighting"
    description: ClassVar[str] = "my-weighted"

def compute_my_weights(log_probs, n_tokens, params):
    # Return normalized weights summing to 1.0
    return [1.0 / len(log_probs)] * len(log_probs)

if ENABLED:
    compute_my_weights = register_method(MyWeightingParams)(compute_my_weights)
```

The method is automatically discovered and applied to all arms.

## Disabling a Weighting Method

To disable a weighting method, set `ENABLED = False` at the top of its file.

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - In-depth specification of the estimation algorithm
