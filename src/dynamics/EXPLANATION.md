# Dynamics Analysis Specification

This document provides the mathematical foundations and implementation details for the dynamics analysis module.

## Configuration

All parameters are in `src/common/default_config.py`:

```python
DYNAMICS_STEP = 4              # Measure every N tokens
DYNAMICS_TRAJS_PER_ARM = 2     # Extremal trajectories per arm
DYNAMICS_ARMS = ["branch"]     # Arm types: "root", "trunk", "branch", "twig"
```

## Overview

The dynamics module tracks how trajectories evolve by scoring partial text at each token position. Three complementary metrics capture different aspects of this evolution:

- **Pull x(k)**: How strong is the normative characterization at position k?
- **Drift y(k)**: How far has the trajectory deviated from its initial state?
- **Potential z(k)**: How far is the trajectory from its final state?

## Mathematical Definitions

### Structure Scores at Position k

At each measurement position k, we:
1. Extract partial text proportional to k/n_tokens
2. Score it using the scoring config to get structure scores

```
scores(k) = Lambda(text[:char_pos])
```

Where:
- `char_pos = int(len(text) * k / n_tokens)` approximates the character position for k tokens
- `Lambda(text)` is the structure compliance vector from scoring

### Pull x(k)

Pull measures the L2 norm of structure scores at position k:

```
x(k) = ||scores(k)||_2 = sqrt(sum_i scores(k)[i]^2)
```

**Interpretation**: Higher pull indicates stronger normative characterization at that point. A trajectory with consistently high pull shows strong normative tendencies throughout.

### Drift y(k)

Drift measures how far the current scores have deviated from the initial scores:

```
y(k) = ||scores(k) - scores(initial)||_2
```

Where `scores(initial)` is the scores at the first measurement position.

**Interpretation**:
- Drift starts at 0 (by definition, initial deviance from initial is zero)
- Rising drift indicates the trajectory is evolving away from its starting state
- Stable drift indicates the trajectory has settled into a consistent pattern

### Potential z(k)

Potential measures how far the current scores are from the final scores:

```
z(k) = ||scores(k) - scores(final)||_2
```

Where `scores(final)` is the scores at the last measurement position (full text).

**Interpretation**:
- Potential ends at 0 (by definition, final deviance from final is zero)
- Decreasing potential indicates the trajectory is converging toward its final state
- High initial potential with rapid decrease suggests late-emerging normative patterns

## Computation Algorithm

### compute_dynamics()

```python
def compute_dynamics(trajectories, config, runner, embedder, step):
    results = []

    for traj_idx, arm_name, text, n_tokens in trajectories:
        # 1. Identify measurement positions
        positions = [step, 2*step, ..., n_tokens]
        if n_tokens not in positions:
            positions.append(n_tokens)

        # 2. Score at each position
        scored = []
        for k in positions:
            ratio = k / n_tokens
            partial_text = text[:int(len(text) * ratio)]
            scores = score_text(partial_text, config, runner, embedder)
            scored.append((k, scores))

        # 3. Compute metrics
        initial_scores = scored[0][1]
        final_scores = scored[-1][1]

        position_scores = []
        for k, scores in scored:
            position_scores.append(PositionScores(
                k=k,
                scores=scores,
                pull=l2_norm(scores),
                drift=deviance(scores, initial_scores),
                potential=deviance(scores, final_scores),
            ))

        results.append(TrajectoryDynamics(...))

    return DynamicsResult(trajectories=results, ...)
```

### Measurement Positions

Measurements are taken at evenly spaced token positions:

```python
def measurement_positions(n_tokens, step):
    positions = list(range(step, n_tokens + 1, step))
    if n_tokens not in positions:
        positions.append(n_tokens)  # Always include final position
    return positions
```

Example with step=4, n_tokens=15:
- Positions: [4, 8, 12, 15]

## Trajectory Selection

Dynamics analyzes **extremal trajectories** - those with highest and lowest inverse perplexity per arm:

```python
def select_extremal(all_trajs, arms_filter, n_per_arm):
    # 1. Filter to configured arm types
    filtered = [t for t in all_trajs if classify_arm(t.arm).value in arms_filter]

    # 2. Group by arm name (e.g., "branch_1", "branch_2")
    # 3. Sort each group by inverse perplexity
    # 4. Pick n_per_arm most extremal, alternating from low/high ends

    for arm_name, group in groupby(sorted(filtered, key=lambda t: t.arm)):
        by_ppl = sorted(group, key=inv_ppl)
        extremal = pick_alternating_ends(by_ppl, n_per_arm)
        yield from extremal
```

**String selection**: The same `STRING_SELECTION` from `default_config.py` is applied before dynamics scoring. For example, `NonThinkingContinuation` strips `<think>...</think>` blocks.

## Visualization

### Output Structure

```
out/<method>/<gen_name>/<scoring_name>/viz/dynamics/
    all/                           # Individual trajectory plots
        traj_0_trunk.png
        traj_1_branch_1.png
    dynamics_trunk.png             # Aggregate: all trunk trajectories
    dynamics_branch_1.png          # Aggregate: all branch_1 trajectories
    dynamics_branch_2.png          # Aggregate: all branch_2 trajectories
```

### Individual Plots (in `all/`)

Each trajectory gets a single plot with three curves:

1. **Pull curve** (orange, triangles): `x(k)` vs token position
2. **Drift curve** (purple, circles): `y(k)` vs token position
3. **Potential curve** (blue, squares): `z(k)` vs token position

### Aggregate Plots

3-column layout (Pull | Drift | Potential) with all trajectories for that arm overlaid. Useful for comparing trajectories within the same arm.

## JSON Output Format

```json
{
  "n_structures": 4,
  "step": 4,
  "trajectories": [
    {
      "traj_idx": 0,
      "arm_name": "trunk",
      "n_tokens": 64,
      "pull": [[4, 1.12], [8, 1.62], [12, 1.45], ...],
      "drift": [[4, 0.0], [8, 0.8], [12, 1.2], ...],
      "potential": [[4, 1.5], [8, 1.2], [12, 0.8], ...]
    }
  ]
}
```

## Interpretation Guide

### Reading Pull Curves

- **High values**: Strong normative characterization at that position
- **Rising pull**: Normative content increasing as text develops
- **Falling pull**: Normative content decreasing (text becoming more neutral)
- **Oscillating pull**: Alternating between normative and neutral content

### Reading Drift Curves

- **Rapid early rise**: Trajectory diverges quickly from its initial state
- **Plateau**: Trajectory has stabilized
- **Continued rise**: Trajectory continues to evolve throughout

### Reading Potential Curves

- **High initial value**: Final state is very different from initial state
- **Gradual decrease**: Smooth convergence to final state
- **Sharp late drop**: Sudden shift to final pattern near the end
- **Zero at end**: By definition, always ends at zero

### Combined Analysis

- **Pull high, drift low**: Strong but stable normative content
- **Pull low, drift high**: Neutral content that varies significantly
- **Potential >> Drift**: Trajectory changes more toward the end
- **Drift >> Potential**: Trajectory changed more at the beginning

## Performance Considerations

- **All metrics require re-scoring**: Each measurement position requires scoring the partial text
- **Step size trade-off**: Smaller step = more resolution but more API calls
- **Typical step=4**: Reasonable balance of resolution and cost
- **Extremal selection**: Only `DYNAMICS_TRAJS_PER_ARM` trajectories per arm are analyzed (not all trajectories)
- **Arm filtering**: Only arm types in `DYNAMICS_ARMS` are analyzed (default: branches only)
