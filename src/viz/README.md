# Visualization Package

Generates comprehensive visualizations of experiment results showing structure compliance (cores), deviance metrics, orientation vectors, and trajectory trees.

## Modules

| Module | Purpose |
|--------|---------|
| `experiment_visualizer.py` | Main entry point: `visualize_result()` and `visualize_generation_comparison()` |
| `experiment_core_barplot.py` | Core stacked bar plots and arm comparisons |
| `experiment_deviance_plot.py` | Deviance trajectories, orientation vectors, and diversity plots |
| `experiment_variants_plot.py` | Generalized cores (heatmaps) and deviance line plots for (q,r) variants |
| `experiment_breakdown_plot.py` | Structure breakdown by branch as grouped horizontal bars |
| `experiment_tree_plot.py` | Trajectory tree visualization with probabilities and structure scores |
| `viz_plot_utils.py` | Shared utilities: colors, axis styling, figure saving |

## Main Functions

### `visualize_result(result, output_dir=None)`
Generates all visualizations for an EstimationResult. Creates per-method plots in subdirectories and cross-method comparison plots.

**Output structure:**
```
out/{method}/{gen_name}/{scoring_name}/viz/
  - core.png                    # Core bar plot
  - deviance.png                # E[∂|self] trajectory
  - excess_deviance.png         # Over-compliance trajectory
  - deficit_deviance.png        # Under-compliance trajectory
  - mutual_deviance.png         # Symmetric deviance trajectory
  - core_diversity.png          # Diversity (D₁) trajectory
  - orientation/{ref}.png       # Orientation bar plot relative to reference arm
  - orientation/evolution_{ref}.png  # Evolution tree relative to reference arm
  - generalized_cores.png       # Heatmap of core variants
  - generalized_deviance.png    # E[∂] line plots as q/r→∞
  - estimation_comparison.png   # Compare cores across weighting methods
  - breakdown.png               # Structure breakdown by branch
  - tree_word.png               # Word-level trajectory tree
  - tree_phrase.png             # Phrase-level trajectory tree

out/{method}/{gen_name}/{scoring_name}/viz/dynamics/
  - drift.png                   # Drift trajectories
  - potential.png                 # Potential deviance plots
```

### `visualize_generation_comparison(results, output_dir=None)`
Compares trunk cores across multiple generation methods. Creates a comparison plot showing grouped bars for each arm across methods. Output is saved to `out/generation_comparisons/{gen_name}/{scoring_name}.png` to prevent overwrites from different experiment configurations.

## Plot Descriptions

**Core Plots**: Stacked horizontal bars showing structure compliance (0-1) per arm. One subplot per arm, stacked vertically. Colors represent different structures.

**Deviance Plots**: Line trajectories showing how deviance metrics evolve through conditioning ancestry (root → trunk → branch → twig). Each arm is a separate line.

**Orientation Plots**: Bar plots showing orientation vectors (signed differences) relative to reference arms. Only generated for arms with downstream children.

**Diversity Plots**: Line trajectories showing Hill D₁ diversity evolution through conditioning stages.

**Generalized Cores**: Heatmaps showing cores across different statistical variants (q,r parameter combinations). Organized by arm rows and variant rows.

**Generalized Deviance**: Dual-axis plots showing E[∂] as q→∞ (r=1 fixed) and r→∞ (q=1 fixed) for each arm.

**Breakdown Plots**: Grouped horizontal bars showing per-branch percentages for all structures (both categorical and bundled questions).

**Tree Plots**: DAG visualization of trajectory branching structure. Node colors reflect structure scores. Edge thickness indicates cumulative log probability. Word-level and phrase-level variants available.

## Shared Utilities

```python
from src.viz.viz_plot_utils import (
    STRUCTURE_COLORS,       # 10-color palette for structures
    get_structure_color,    # Get color by index
    style_axis_clean,       # Clean axis styling
    annotate_bar_values,    # Add value labels on bars
    save_figure,            # Save and close figure
    lighten_color,          # Lighten hex color for blending
    add_reference_line,     # Add horizontal reference lines
)
```

Arm colors are provided by `src.estimation.arm_types.get_arm_color(arm_name)`.
