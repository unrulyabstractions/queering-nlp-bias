# Dynamics Module

Analyze how trajectories evolve by scoring partial text at each token position.

## Configuration

Parameters in `src/common/default_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DYNAMICS_STEP` | 4 | Measure scores every N tokens |
| `DYNAMICS_TRAJS_PER_ARM` | 2 | Most extremal trajectories to analyze per arm |
| `DYNAMICS_ARMS` | `["branch"]` | Arm types to analyze: `"root"`, `"trunk"`, `"branch"`, `"twig"` |

## Directory Structure

```
dynamics/
├── dynamics_types.py               # PositionScores, TrajectoryDynamics, DynamicsResult
├── dynamics_computation.py         # compute_dynamics() - main computation
├── dynamics_visualization.py       # plot_dynamics() - per-trajectory plots
└── dynamics_serialization.py       # save_dynamics_json() - save to JSON
```

## Metrics

At each token position k, we score the partial text and compute three metrics:

### Pull x(k)

L2 norm of the structure scores at position k:
```
pull(k) = ||scores(k)||_2
```
Represents the "strength" of normative characterization at that point.

### Drift y(k)

Deviance of scores at position k from the initial scores (at the first measured position):
```
drift(k) = ||scores(k) - scores(initial)||_2
```
Measures how far the trajectory has evolved from its starting point.

### Potential z(k)

Deviance of scores at position k from the final scores:
```
potential(k) = ||scores(k) - scores(final)||_2
```
Measures how far the trajectory is from its end state.

## Trajectory Selection

Dynamics analyzes **extremal trajectories** (highest and lowest inverse perplexity per arm):

1. Filter to arm types in `DYNAMICS_ARMS`
2. Group by arm name (e.g., `branch_1`, `branch_2`)
3. Sort by inverse perplexity
4. Pick `DYNAMICS_TRAJS_PER_ARM` most extremal, alternating from low/high ends

**String selection**: Same as scoring - `STRING_SELECTION` is applied (e.g., strips `<think>...</think>` blocks).

## Output Structure

Dynamics outputs are saved to:
```
out/<method>/<gen_name>/<scoring_name>/
    dynamics.json                    # JSON with (k, value) pairs
    viz/dynamics/
        all/                         # Individual trajectory plots
            traj_0_trunk.png
            traj_1_branch_1.png
        dynamics_trunk.png           # Aggregate: all trunk trajectories overlaid
        dynamics_branch_1.png        # Aggregate: all branch_1 trajectories (3 columns)
        dynamics_branch_2.png        # Aggregate: all branch_2 trajectories (3 columns)
```

**Individual plots** (in `all/`): Single figure with pull, drift, potential curves.

**Aggregate plots**: 3-column layout (Pull | Drift | Potential) with all trajectories for that arm overlaid.

Each plot shows three curves:
- **Pull** (orange): L2 norm of scores at each position
- **Drift** (purple): deviance from initial scores
- **Potential** (blue): deviance from final scores

## Quick Start

```python
from src.dynamics import compute_dynamics, plot_dynamics, save_dynamics_json
from src.estimation import ScoringData

# Load trajectory data
scoring_data = ScoringData.load("out/.../scoring.json")
trajectories = [
    (t.traj_idx, t.arm, t.text, t.n_generated_tokens)
    for t in scoring_data.get_all_trajectories()
]

# Compute dynamics
result = compute_dynamics(
    trajectories=trajectories,
    config=scoring_config,
    runner=model_runner,       # For LLM-based scoring
    embedder=embedding_runner, # For embedding-based scoring
    step=4,                    # Measure every 4 tokens
)

# Save and plot
save_dynamics_json(result, Path("out/.../dynamics.json"))
saved_paths = plot_dynamics(result, Path("out/.../viz/dynamics/"))
```

## Key Types

| Type | Description |
|------|-------------|
| `PositionScores` | Scores and metrics at a specific token position (k, scores, pull, drift, potential) |
| `TrajectoryDynamics` | All dynamics data for one trajectory |
| `DynamicsResult` | Result for all analyzed trajectories |

## See Also

- [CLAUDE.md](./CLAUDE.md) - Quick reference
- [../estimation/CLAUDE.md](../estimation/CLAUDE.md) - Estimation module
