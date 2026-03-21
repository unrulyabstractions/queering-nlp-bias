# CLAUDE.md - src/viz/

This module generates matplotlib visualizations of experiment results.

## Main Entry Points

```python
from src.viz import visualize_result, visualize_generation_comparison

# Full visualization suite for one experiment
visualize_result(estimation_result, output_dir=Path("out/.../viz/"))

# Compare trunk cores across generation methods
visualize_generation_comparison(results_list, output_dir=Path("out/.../"))
```

## Module Structure

| File | Purpose |
|------|---------|
| `experiment_visualizer.py` | Main entry: `visualize_result()`, `visualize_generation_comparison()` |
| `experiment_core_barplot.py` | Stacked bar plots for structure compliance |
| `experiment_deviance_plot.py` | Deviance trajectories, orientation, diversity |
| `experiment_variants_plot.py` | Generalized cores heatmaps, (q,r) variant plots |
| `experiment_breakdown_plot.py` | Per-branch structure breakdown |
| `experiment_tree_plot.py` | Trajectory tree DAG visualization |
| `viz_plot_utils.py` | **Shared utilities** - use these, don't duplicate |

## Shared Utilities (viz_plot_utils.py)

**Always check here before writing new plotting code.**

```python
from src.viz.viz_plot_utils import (
    STRUCTURE_COLORS,       # 10-color palette for structures
    get_structure_color,    # Get color by index
    style_axis_clean,       # Remove spines, add grid
    annotate_bar_values,    # Add value labels on bars
    save_figure,            # Save PNG and close
    lighten_color,          # Lighten hex color
    add_reference_line,     # Horizontal reference lines
)
```

## Arm Colors

Arm colors come from `src.estimation.arm_types`:

```python
from src.estimation.arm_types import get_arm_color

color = get_arm_color("trunk")  # Returns hex like "#E67E22"
```

## Output Structure

```
out/<method>/<gen_name>/<scoring_name>/viz/
    core.png                  # Structure compliance bars
    deviance.png              # E[d|self] trajectory
    generalized_cores.png     # (q,r) variant heatmap
    breakdown.png             # Per-branch structure breakdown
    tree_word.png             # Word-level trajectory tree
    tree_phrase.png           # Phrase-level tree
    orientation/
        trunk.png             # Orientation relative to trunk
        evolution_root.png    # Evolution tree from root
        evolution_trunk.png   # Evolution tree from trunk

out/<method>/<gen_name>/<scoring_name>/viz/dynamics/
    traj_0_trunk.png          # Per-trajectory dynamics
    ...

out/generation_comparisons/<gen_name>/
    <scoring_name>.png        # Compare methods (multi-method runs)
```

## Plot Types

### Core Plots
Stacked horizontal bars showing structure compliance (0-1) per arm.

### Deviance Plots
Line trajectories showing metric evolution through arm ancestry (root -> trunk -> branch -> twig).

### Orientation Plots
Signed bar plots showing orientation vectors relative to reference arms.

### Tree Plots
DAG visualization with:
- Node colors = structure scores
- Edge thickness = cumulative log probability
- Word-level and phrase-level variants

## Common Pitfalls

1. **Use `save_figure()`** - handles directory creation and closes figure
2. **Use `style_axis_clean()`** - consistent axis styling
3. **Get arm colors from arm_types** - not hardcoded
4. **Use STRUCTURE_COLORS palette** - for structure indices

## Adding New Plots

1. Create `experiment_myplot.py` in this directory
2. Import utilities from `viz_plot_utils.py`
3. Add call to `experiment_visualizer.py`

Follow the pattern:
```python
def plot_my_chart(result, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    # ... plotting code using shared utilities ...
    return save_figure(fig, output_dir / "my_chart.png")
```

## See Also

- [README.md](./README.md) - module overview and plot descriptions
- `src.common.viz/tree_display.py` - ASCII tree visualization (different from matplotlib)
- [Root CLAUDE.md](../../CLAUDE.md) - global project rules

## Workflow Orchestration

1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
