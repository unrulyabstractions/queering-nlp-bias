# Dynamics Analysis Specification

This document provides the mathematical foundations and implementation details for the dynamics analysis module.

## Configuration

All parameters are in `src/common/default_config.py`:

```python
DYNAMICS_STEP = 8                  # Measure the system default every N tokens
DYNAMICS_SAMPLES_PER_POSITION = 8  # Continuations sampled per prefix for the barycenter
DYNAMICS_CONTINUATION_MAX_TOKENS = 128  # Tokens per sampled continuation
DYNAMICS_TRAJS_PER_ARM = 2         # Extremal trajectories per arm
DYNAMICS_ARMS = ["root", "trunk", "branch", "twig"]  # Arm types analyzed
```

## Overview

The dynamics module tracks how a trajectory evolves token by token. At each measured
position it tracks two distinct paper quantities — the realized **system attunement**
Λ_n(x_p) and the **system default** ⟨Λ_n⟩(x_p) (the barycenter) — and from them computes
three deviance-based metrics:

- **Pull x(k)**: How strong is the normative attractor (the system default) at position k?
- **Drift y(k)**: How far has the realized text drifted from the *initial* normative frame?
- **Potential z(k)**: How far is the final outcome from the *current* normative frame?

## Mathematical Definitions

### System Attunement and System Default at Position k

At each measurement position k we compute two distinct vectors, never conflated:

1. **System attunement** Λ_n(x_p^k) — score the prefix text directly (paper Eqs. 2-3):
   ```
   Λ_n(x_p^k) = (α_1(x_p^k), ..., α_n(x_p^k))
   ```
   where `char_pos = int(len(text) * k / n_tokens)` approximates the character position
   for k tokens.

2. **System default** ⟨Λ_n⟩(x_p^k) — the expected attunement over continuations of the
   prefix (paper Eq. 7), estimated by Monte-Carlo sampling:
   ```
   ⟨Λ_n⟩(x_p^k) ≈ (1/M) Σ_{m=1..M} Λ_n(y_m),   y_m ~ P(· | x_p^k)
   ```
   We sample M = `samples_per_position` completions from the model at temperature 1.0
   (so the uniform mean is an unbiased estimate of the expectation), score each, average.

All metrics use the paper's **dimension-normalized** norms,
`||v||_Λ = ||v||_θ = ||v||_2 / sqrt(dim)` (paper Eqs. 4, 5, 9), so they stay in `[0, 1]`
when scores are in `[0, 1]`. Each metric is an **orientation/deviance**: an *attunement*
of a string minus a *system default* of a reference prefix.

### Pull x(k)

Pull measures the dimension-normalized magnitude of the **system default** at position k:

```
x(k) = ||⟨Λ_n⟩(x_p^k)||_Λ
```

**Interpretation**: higher pull = a more strongly oriented normative attractor at that point.

### Drift y(k)

Drift is the deviance of the current **attunement** from the **initial system default**:

```
y(k) = ∂_n(x_p^k | x_0) = ||Λ_n(x_p^k) - ⟨Λ_n⟩(x_0)||_θ
```

where `⟨Λ_n⟩(x_0)` is the system default at the first measurement position.

**Interpretation**:
- Drift does NOT start at 0: at k=0 it is `||Λ_n(x_0) - ⟨Λ_n⟩(x_0)||`, the realized
  prefix's deviance from its own default (generally nonzero).
- Rising drift indicates the realized text is moving away from the starting normative frame.

### Potential z(k)

Potential is the deviance of the **final attunement** from the **current system default**:

```
z(k) = ∂_n(x_final | x_p^k) = ||Λ_n(x_final) - ⟨Λ_n⟩(x_p^k)||_θ
```

where `Λ_n(x_final)` is the attunement at the last measurement position (full text).

**Interpretation**:
- Potential does NOT end at 0: at the last position it is the final attunement's deviance
  from the final prefix's default (generally nonzero).
- Decreasing potential indicates the system default is converging toward the realized end.

## Computation Algorithm

### compute_dynamics()

```python
def compute_dynamics(trajectories, scorer, runner, config):
    results = []

    for traj in trajectories:  # DynamicsTrajectory: prompt, prefill, text, n_tokens, ...
        # 1. Identify measurement positions
        positions = [step, 2*step, ..., n_tokens]
        if n_tokens not in positions:
            positions.append(n_tokens)

        # 2. At each position: realized attunement + SAMPLED system default
        measured = []
        for k in positions:
            prefix_text = traj.text[:int(len(traj.text) * k / traj.n_tokens)]
            attunement = scorer.score(prefix_text)                      # Λ_n(x_p)
            default = estimate_system_default(                          # ⟨Λ_n⟩(x_p)
                runner, scorer, traj.prompt, traj.prefill, prefix_text,
                config.samples_per_position, config.continuation_max_tokens, config.temperature,
            )  # samples M continuations, scores each, averages
            measured.append((k, attunement, default))

        # 3. Compute deviance-based metrics (normalized_deviance = ||a-b||_2/sqrt(n))
        initial_default = measured[0][2]     # ⟨Λ_n⟩(x_0)
        final_attunement = measured[-1][1]   # Λ_n(x_final)

        position_scores = []
        for k, attunement, default in measured:
            position_scores.append(PositionScores(
                k=k,
                system_attunement=attunement,
                system_default=default,
                pull=pull(default),
                drift=drift(attunement, initial_default),
                potential=potential(final_attunement, default),
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

Example with step=8, n_tokens=20:
- Positions: [8, 16, 20]

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
  "step": 8,
  "trajectories": [
    {
      "traj_idx": 0,
      "arm_name": "trunk",
      "n_tokens": 64,
      "pull": [[8, 0.42], [16, 0.53], [24, 0.64], ...],
      "drift": [[8, 0.03], [16, 0.11], [24, 0.25], ...],
      "potential": [[8, 0.39], [16, 0.25], [24, 0.12], ...]
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

- **High initial value**: the final outcome is far from the early normative frame
- **Gradual decrease**: the system default converges toward the realized end
- **Sharp late drop**: sudden shift to the final pattern near the end
- **Small (not zero) at end**: the final attunement still deviates from the final default

### Combined Analysis

- **Pull high, drift low**: Strong but stable normative content
- **Pull low, drift high**: Neutral content that varies significantly
- **Potential >> Drift**: Trajectory changes more toward the end
- **Drift >> Potential**: Trajectory changed more at the beginning

## Performance Considerations

- **Sampling dominates cost**: each measured position samples `samples_per_position`
  continuations from the model to estimate the system default. Generations per trajectory
  ≈ `positions × samples_per_position` (e.g. step=8 over 128 tokens × 8 samples ≈ 128).
- **Two models resident**: the judge model (scorer) and the generation model (runner) are
  loaded together during dynamics — watch GPU memory.
- **Step / sample trade-off**: raise `DYNAMICS_STEP` or lower `DYNAMICS_SAMPLES_PER_POSITION`
  for speed; lower step / more samples for resolution and a lower-variance default.
- **temperature=1.0**: required for the uniform sample mean to be a valid estimate of ⟨Λ_n⟩.
- **Extremal selection**: only `DYNAMICS_TRAJS_PER_ARM` trajectories per arm are analyzed.
- **Arm filtering**: only arm types in `DYNAMICS_ARMS` are analyzed.
