# Dynamics Module

Analyze how trajectories evolve relative to reference cores through drift, horizon, and pull measurements.

## Directory Structure

```
dynamics/
├── logging/                        # Logging utilities
│   └── dynamics_step_logging.py      # Step-by-step drift/horizon/pull logging
├── dynamics_types.py               # DriftPoint, HorizonPoint, PullPoint, TrajectoryDynamics
├── dynamics_computation.py         # Compute dynamics metrics
└── dynamics_visualization.py       # Plot drift/horizon/pull curves
```

## Metrics

### Drift y(k)

Deviance of **partial text** (re-scored at token position k) relative to the **root core**.

- Re-runs scoring on partial text at multiple token positions
- Measures how far the trajectory has deviated from root as text develops
- Plotted as a continuous curve over token position

### Horizon z(arm)

Deviance of **full trajectory** relative to each **arm's core** along the trajectory's ancestry path.

- Uses pre-computed full trajectory scores (no re-scoring)
- Only computes for arms on this trajectory's conditioning path (see Arm Ancestry below)
- Plotted at each arm's prefix token count (the length of text up to that arm)

### Pull x(arm)

L2 norm of each **arm's core** vector, plotted at the arm's prefix position.

- Represents the "strength" of normative characterization at each arm
- Higher pull = stronger normative tendencies at that conditioning point
- Plotted alongside horizon for comparison

## Arm Ancestry

A trajectory only computes horizon and pull for arms on its **ancestry path**. The ancestry is determined by the conditioning hierarchy:

```
root -> trunk -> branch_N -> twig_M_bN
```

Examples:
- `branch_1` trajectory: computes for `["root", "trunk", "branch_1"]`
- `branch_2` trajectory: computes for `["root", "trunk", "branch_2"]`
- `twig_2_b1` trajectory: computes for `["root", "trunk", "branch_1", "twig_2_b1"]`

Use `get_arm_ancestry(arm_name)` from `arm_types.py` to get the ancestry path.

## Output Structure

Dynamics plots are saved to:
```
out/<method>/<gen_name>/<scoring_name>/viz/dynamics/traj_{idx}_{arm}.png
```

Each plot shows three curves:
- **Drift** (purple): deviance from root core as text develops
- **Horizon** (blue): deviance from each arm's core at arm prefix positions
- **Pull** (orange): L2 norm of arm's core at arm prefix positions

## Quick Start

```python
from src.estimation.dynamics import compute_dynamics, plot_dynamics

# Compute dynamics for representative trajectories
result = compute_dynamics(
    estimation_result=estimation_result,
    scoring_config=scoring_config,
    runner=model_runner,       # For LLM-based scoring
    embedder=embedding_runner, # For embedding-based scoring
    trajs_per_arm=1,           # Analyze 1 trajectory per arm
)

# Generate plots
output_dir = Path("out/simple-sampling/gen_name/score_name/viz/dynamics")
saved_paths = plot_dynamics(result, output_dir)
```

## Key Types

| Type | Description |
|------|-------------|
| `DriftPoint` | Drift measurement at a token position (partial scores, deviance) |
| `HorizonPoint` | Horizon measurement at an arm (arm name, prefix tokens, deviance) |
| `PullPoint` | Pull measurement at an arm (arm name, prefix tokens, L2 norm) |
| `TrajectoryDynamics` | All dynamics data for one trajectory |
| `DynamicsResult` | Result for all analyzed trajectories |

## See Also

- [../EXPLANATION.md](../EXPLANATION.md) - Full estimation algorithm specification
- [../arm_types.py](../arm_types.py) - Arm classification and ancestry utilities
