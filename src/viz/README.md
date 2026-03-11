# Visualization Package

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.

Visualization tools for experiment results. Generates plots showing:
1. **Core bar plots** - Structure compliance cores across arms (trunk vs branches)
2. **Tree plots** - Trajectory branching structure with probabilities and scores

## Directory Structure

```
viz/
├── experiment_visualizer.py    # Main entry point: visualize_result()
├── experiment_core_barplot.py  # Bar plots comparing cores across arms
└── experiment_tree_plot.py     # Tree visualization for trajectories
```

## Quick Start

```python
from src.viz import visualize_result
from src.estimation.estimation_experiment_types import EstimationResult

# After running an experiment, visualize the result
result = EstimationResult.from_estimation_file(method_name, paths)
visualize_result(result)  # Saves to out/viz/
```

## Output Files

For each experiment, the following files are generated in `out/viz/`:

### Core Plots
| File Pattern | Description |
|--------------|-------------|
| `core_{method}_{weighting}.png` | Stacked bar plot (one row per arm) for each weighting method |
| `core_{method}_comparison.png` | Comparison plot with arms side-by-side (one row per weighting method) |

### Deviance & Orientation Plots
| File Pattern | Description |
|--------------|-------------|
| `deviance_{method}_{weighting}.png` | Line plot showing E[∂\|trunk] → E[∂\|branch] for each branch |
| `deviance_{method}_delta.png` | Bar plot showing Δ∂ (branch deviance - trunk deviance) |
| `orientation_{method}_by_branch.png` | Bar plot showing E[θ\|T] per structure for each branch |

### Tree Plots
| File Pattern | Description |
|--------------|-------------|
| `tree_{method}_word.png` | Word-level trajectory tree with curved edges |
| `tree_{method}_phrase.png` | Collapsed phrase-level trajectory tree |

## Core Bar Plots

Shows structure compliance cores as grouped bars:
- X-axis: Structure labels (c1, c2, g1, s1, etc.)
- Y-axis: Core value [0, 1]
- One bar group per arm (trunk, branch_1, etc.)
- One plot per weighting method (prob, inv-ppl, uniform)

```python
from src.viz.experiment_core_barplot import create_core_barplots

files = create_core_barplots(result, structure_labels, output_dir)
```

## Tree Plots

Visualizes trajectory branching structure:
- Nodes represent words/phrases in continuations
- Edge thickness proportional to probability
- Node colors indicate dominant structure
- Leaf nodes show structure scores

Two modes:
- **Word tree**: Every word is a node
- **Phrase tree**: Single-child chains collapsed into phrases

```python
from src.viz.experiment_tree_plot import create_tree_plots

files = create_tree_plots(result, output_dir)
```

## Integration with Experiments

The `visualize_result()` function is automatically called in `run_single_experiment()`:

```python
# In scripts/run_full_experiment.py
def run_single_experiment(...) -> EstimationResult:
    # ... run pipeline ...
    result = EstimationResult.from_estimation_file(...)
    visualize_result(result)  # Auto-generates plots
    return result
```

## Customization

### Custom Output Directory

```python
visualize_result(result, output_dir="my_plots/")
```

### Structure Labels

Labels are automatically loaded from the scoring output. If not available,
generic labels (s1, s2, ...) are generated based on core vector length.
