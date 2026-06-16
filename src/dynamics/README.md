# Dynamics Module

Track how a trajectory evolves token by token, computing the paper's **deviance-based**
pull/drift/potential at each position. At every measured prefix it tracks two distinct
quantities:

- **system attunement** Λ_n(x_p) — score of the realized prefix text, and
- **system default** ⟨Λ_n⟩(x_p) — the barycenter, estimated by **sampling continuations**
  from the model at that prefix and averaging their attunements (paper Eq. 7).

## Configuration

Parameters in `src/common/default_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DYNAMICS_STEP` | 8 | Measure the system default every N tokens (each measure samples, so small = expensive) |
| `DYNAMICS_SAMPLES_PER_POSITION` | 8 | Continuations sampled per prefix to estimate ⟨Λ_n⟩(x_p) |
| `DYNAMICS_CONTINUATION_MAX_TOKENS` | 128 | Max tokens per sampled continuation |
| `DYNAMICS_TRAJS_PER_ARM` | 2 | Most extremal trajectories to analyze per arm |
| `DYNAMICS_ARMS` | all four | Arm types to analyze: `"root"`, `"trunk"`, `"branch"`, `"twig"` |

## Directory Structure

```
dynamics/
├── dynamics_types.py            # DynamicsTrajectory, DynamicsConfig, PositionScores, TrajectoryDynamics, DynamicsResult
├── dynamics_metrics.py          # pull(), drift(), potential(), normalized_norm()
├── dynamics_sampling.py         # estimate_system_default() — samples continuations → barycenter
├── dynamics_computation.py      # compute_dynamics() - main computation
├── dynamics_visualization.py    # plot_dynamics() - per-trajectory plots
└── dynamics_serialization.py    # save_dynamics_json() - save to JSON
```

## Metrics

Every metric is an **orientation/deviance** (paper Eqs. 8-9): an *attunement* of a string
minus a *system default* of a reference prefix — never attunement−attunement or
default−default. All use the dimension-normalized norm `||·|| = ||·||_2 / sqrt(dim)`, so
values land in `[0, 1]` for scores in `[0, 1]`.

### Pull x(k)
```
pull(k) = ||⟨Λ_n⟩(x_p)||
```
Magnitude of the **system default** at the prefix — strength of the normative attractor.

### Drift y(k)
```
drift(k) = ||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||
```
Deviance of the current **attunement** from the **initial system default** — how far the
realized text has drifted from the starting normative frame.

### Potential z(k)
```
potential(k) = ||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||
```
Deviance of the **final attunement** from the **current system default** — its remaining pull.

## Trajectory Selection

Dynamics analyzes **extremal trajectories** (highest and lowest inverse perplexity per arm):

1. Filter to arm types in `DYNAMICS_ARMS`
2. Group by arm name (e.g., `branch_1`, `branch_2`)
3. Sort by inverse perplexity
4. Pick `DYNAMICS_TRAJS_PER_ARM` most extremal, alternating from low/high ends

**String selection**: the same `STRING_SELECTION` is applied (e.g. `NonThinkingContinuation`
strips `<think>...</think>` blocks) to both the analyzed text and each sampled continuation.

## Quick Start

```python
from pathlib import Path
from src.dynamics import (
    DynamicsConfig, DynamicsTrajectory,
    compute_dynamics, plot_dynamics, save_dynamics_json,
)
from src.inference import ModelRunner
from src.scoring import Scorer

scorer = Scorer.load(scoring_config_path)      # judge model → system attunement
runner = ModelRunner(gen_model_name)           # generation model → sampled continuations

# Each trajectory carries the prompt + arm prefill needed to sample continuations.
trajectories = [
    DynamicsTrajectory(
        traj_idx=t.traj_idx, arm_name=t.arm, prompt=prompt, prefill=arm_prefill,
        text=continuation, n_tokens=t.n_generated_tokens,
    )
    for t in selected_trajectories
]

config = DynamicsConfig(step=8, samples_per_position=8, continuation_max_tokens=128, temperature=1.0)
result = compute_dynamics(trajectories, scorer, runner, config)

save_dynamics_json(result, Path("out/.../dynamics.json"))
plot_dynamics(result, Path("out/.../viz/dynamics/"))
```

> **Cost:** generation runs `positions × samples_per_position` times per trajectory. With
> `step=8`, `samples_per_position=8`, a 128-token trajectory ≈ 16 × 8 = 128 generations.
> Raise `step` / lower `samples_per_position` to trade resolution for speed.

## Key Types

| Type | Description |
|------|-------------|
| `DynamicsTrajectory` | A trajectory to analyze: traj_idx, arm_name, prompt, prefill, text, n_tokens |
| `DynamicsConfig` | step, samples_per_position, continuation_max_tokens, temperature |
| `PositionScores` | Per position: k, system_attunement, system_default, pull, drift, potential |
| `TrajectoryDynamics` | All dynamics data for one trajectory |
| `DynamicsResult` | Result for all analyzed trajectories |

## See Also

- [CLAUDE.md](./CLAUDE.md) - Quick reference
- [EXPLANATION.md](./EXPLANATION.md) - Mathematical specification
- [../estimation/CLAUDE.md](../estimation/CLAUDE.md) - Estimation module
