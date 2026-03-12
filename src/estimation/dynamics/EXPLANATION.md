# Dynamics Analysis Specification

This document provides the mathematical foundations and implementation details for the dynamics analysis module.

## Overview

The dynamics module tracks how trajectories evolve relative to reference cores as text develops. Three complementary metrics capture different aspects of this evolution:

- **Drift y(k)**: How far has the partial text deviated from the root at position k?
- **Horizon z(arm)**: How far is the complete trajectory from each arm's normative core?
- **Pull x(arm)**: How strong is each arm's normative characterization?

## Mathematical Definitions

### Drift y(k)

Drift measures the deviance of partial text at token position k, relative to the root core:

```
y(k) = deviance(Lambda_n(text[:k]), core_root)
     = ||orientation(Lambda_n(text[:k]), core_root)||_2
```

Where:
- `text[:k]` is the trajectory text truncated to approximately k tokens
- `Lambda_n(text[:k])` is the structure compliance vector from re-scoring the partial text
- `core_root` is the root arm's core (expected compliance under prompt-only conditioning)
- `orientation(scores, core) = scores - core`
- `deviance(theta, zero) = ||theta||_2`

**Key insight**: Drift requires re-scoring partial text, making it computationally expensive but revealing when non-normative behavior emerges.

### Horizon z(arm)

Horizon measures the deviance of the full trajectory relative to each arm's core along the trajectory's ancestry path:

```
z(arm) = deviance(Lambda_n(full_text), core_arm)
       = ||orientation(Lambda_n(full_text), core_arm)||_2
```

Where:
- `full_text` is the complete trajectory text
- `Lambda_n(full_text)` uses pre-computed scores (no re-scoring needed)
- `core_arm` is the arm's core vector

**Key insight**: Horizon uses pre-computed scores, making it cheap. The horizon relative to the trajectory's own arm indicates how atypical this trajectory is within its conditioning context.

### Pull x(arm)

Pull measures the L2 norm of each arm's core, representing the strength of normative tendencies:

```
x(arm) = ||core_arm||_2 = sqrt(sum_i core_arm[i]^2)
```

**Key insight**: Higher pull indicates stronger normative characterization. Arms with high pull exert stronger "normative force" on trajectories.

## Arm Ancestry and Conditioning Hierarchy

Trajectories are generated under specific conditioning contexts. The **ancestry path** traces back through the conditioning hierarchy:

```
root (prompt only)
  |
  v
trunk (prompt + trunk text)
  |
  v
branch_N (prompt + trunk + branch text)
  |
  v
twig_M_bN (prompt + trunk + branch + twig text)
```

A trajectory only computes horizon and pull for arms **on its ancestry path**. This reflects the fact that a trajectory conditioned on `branch_1` was never exposed to `branch_2`'s conditioning context.

### Ancestry Examples

| Trajectory Arm | Ancestry Path |
|----------------|---------------|
| `root` | `["root"]` |
| `trunk` | `["root", "trunk"]` |
| `branch_1` | `["root", "trunk", "branch_1"]` |
| `branch_2` | `["root", "trunk", "branch_2"]` |
| `twig_2_b1` | `["root", "trunk", "branch_1", "twig_2_b1"]` |

### Prefix Token Estimation

Each arm is associated with a prefix token count (the approximate number of tokens in the conditioning text up to that arm):

```
prefix_tokens(root) = 0
prefix_tokens(trunk) = tokens(trunk_text)
prefix_tokens(branch_N) = tokens(trunk_text + branch_N_text)
prefix_tokens(twig_M_bN) = tokens(trunk_text + branch_N_text + twig_M_text)
```

Token counts are estimated as `len(text) // 4` (roughly 4 characters per token).

## Computation Algorithm

### compute_dynamics()

```python
def compute_dynamics(estimation_result, scoring_config, runner, embedder, trajs_per_arm):
    # 1. Load scoring data and arm cores
    scoring_data = ScoringData.load(estimation_result.paths.judgment)
    arm_cores = {arm.name: arm.get_core("prob") for arm in estimation_result.arms}
    root_core = arm_cores["root"]

    # 2. Estimate arm prefix token counts
    arm_prefix_tokens = {arm_name: estimate_tokens(arm_text) for ...}
    arm_prefix_tokens["root"] = 0

    # 3. Select representative trajectories (trajs_per_arm from each arm)
    selected = select_representative_trajectories(trajectories_by_arm, trajs_per_arm)

    # 4. Compute dynamics for each trajectory
    for traj in selected:
        # Drift: re-score partial text at multiple positions
        for k in measurement_positions:
            partial_scores = score_partial_text(traj.text[:char_pos])
            drift_deviance = compute_deviance(partial_scores, root_core)
            drift_points.append(DriftPoint(k, partial_scores, drift_deviance))

        # Horizon & Pull: only for arms on this trajectory's ancestry path
        ancestry = get_arm_ancestry(traj.branch)
        for arm_name in ancestry:
            arm_core = arm_cores[arm_name]
            prefix_tokens = arm_prefix_tokens[arm_name]

            # Horizon: deviance of full trajectory from arm's core
            horizon_deviance = compute_deviance(traj.full_scores, arm_core)
            horizon_points.append(HorizonPoint(arm_name, prefix_tokens, horizon_deviance))

            # Pull: L2 norm of arm's core
            pull_value = l2_norm(arm_core)
            pull_points.append(PullPoint(arm_name, prefix_tokens, pull_value))
```

### Measurement Positions

Drift measurements are taken at evenly spaced token positions:

```python
def compute_measurement_positions(n_tokens, min_points):
    step = max(1, n_tokens // min_points)
    positions = range(step, n_tokens + 1, step)
    if n_tokens not in positions:
        positions.append(n_tokens)  # Always include final position
    return positions
```

The number of drift points is at least `2 * n_arms`, ensuring sufficient resolution to see dynamics.

## Visualization

### Plot Structure

Each trajectory gets a single plot with three curves:

1. **Drift curve** (purple, circles): `y` vs token position
2. **Horizon points** (blue, squares): `z` vs arm prefix position, connected as a line
3. **Pull points** (orange, triangles): `x` vs arm prefix position, connected as a line

### Output Paths

```
out/<method>/<gen_name>/<scoring_name>/viz/dynamics/traj_{idx}_{arm}.png
```

Example:
```
out/simple-sampling/gen_example/score_example/viz/dynamics/traj_0_branch_1.png
out/simple-sampling/gen_example/score_example/viz/dynamics/traj_1_twig_2_b1.png
```

## Interpretation

### Reading Drift Curves

- **Rapid early rise**: Trajectory diverges quickly from root conditioning
- **Plateau**: Trajectory has stabilized relative to root
- **Oscillation**: Trajectory alternates between normative and non-normative content

### Reading Horizon Points

- **Low deviance at own arm**: Trajectory is typical for its conditioning context
- **High deviance at own arm**: Trajectory is atypical/non-normative within its arm
- **Decreasing along ancestry**: Trajectory becomes more normative at deeper conditioning

### Reading Pull Points

- **High pull at early arms**: Strong normative characterization emerges early
- **Increasing pull along ancestry**: Conditioning becomes more specific/normative
- **Low pull throughout**: Weak normative tendencies, high diversity expected

## Performance Considerations

- **Drift is expensive**: Requires re-scoring at each measurement position
- **Horizon/Pull are cheap**: Use pre-computed scores and cores
- **Limit trajs_per_arm**: Each additional trajectory multiplies drift scoring calls
- **Measurement positions**: `min_drift_points = 2 * n_arms` balances resolution and cost
