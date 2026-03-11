# Estimation Package

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


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
├── weighting_method_registry.py      # Registry pattern for weighting methods
├── estimation_pipeline.py            # Main estimation pipeline
├── estimation_output.py              # Output structure and serialization
├── estimation_scoring_data.py        # Input data loading (ScoringData)
├── estimation_structure.py           # ArmEstimate, TrajectoryScoringData types
├── estimation_weighted_types.py      # WeightedEstimate type
├── estimation_core_types.py          # CoreVariant, NAMED_CORES (q,r) params
├── estimation_auxiliary_types.py     # Summary and continuation helper types
├── estimation_scoring_result.py      # StructureInfo, ArmScoring types
└── estimation_experiment_types.py    # EstimationResult for comparisons
```

## Quick Start

```python
from src.estimation import ScoringData, run_estimation_pipeline

# Load scoring data
scoring_data = ScoringData.load("out/score_simple-sampling_example_example.json")

# Run estimation
result = run_estimation_pipeline(scoring_data, "out/score_simple-sampling_example_example.json")

# Access results - iterate over weighting methods
for arm in result.arms:
    for method_name, estimate in arm.estimates.items():
        print(f"{arm.name} [{method_name}]: core={estimate.core}, E[d]={estimate.deviance_avg}")
```

## Key Files

| File | Purpose |
|------|---------|
| `estimation_pipeline.py` | Core algorithm: `run_estimation_pipeline()`, `compute_arm_estimate()` |
| `weighting_method_registry.py` | Registry for pluggable weighting methods |
| `estimation_scoring_data.py` | Load and process scoring JSON files |
| `estimation_weighted_types.py` | `WeightedEstimate` - results for one weighting method |
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
