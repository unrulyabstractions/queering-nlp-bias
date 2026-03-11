# Estimation Methodology

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

How we compute normativity metrics from scored trajectories.

## Conceptual Framework

Given an LLM and a system of structures, we can reason about **expected structural compliance**. The **system core** is:

```
⟨Λ_n⟩ = Σ_{y ∈ Str⊤} p(y) Λ_n(y)
```

The core characterizes **normativity** - what the model treats as the default pattern.

The **orientation** of a trajectory relative to the core is:

```
θ_n(x) = Λ_n(x) - ⟨Λ_n⟩
```

And the **deviance** is the magnitude:

```
∂_n(x) = ||θ_n(x)||
```

## Core Computation

Since we can't sum over all possible trajectories, we approximate using our samples:

```python
core = Σ_i w_i * Λ_n(x_i)
```

Where `w_i` are normalized weights derived from trajectory probabilities.

### Probability-Weighted Core

Weights are trajectory probabilities, normalized:

```python
log_probs = [traj.conditional_logprob for traj in trajectories]
max_lp = max(log_probs)
probs = [exp(lp - max_lp) for lp in log_probs]
weights = [p / sum(probs) for p in probs]
```

### Inverse-Perplexity-Weighted Core

Weights are inverse perplexity (model confidence per token):

```python
inv_ppls = [exp(logprob / n_tokens) for traj in trajectories]
weights = [p / sum(inv_ppls) for p in inv_ppls]
```

This weights by average token confidence rather than total probability.

## Generalized Cores

### Why a Profile, Not a Single Number

A fundamental insight from diversity theory (Leinster, *Entropy and Diversity*, 2021): **diversity cannot be captured by a single number**. Instead, it requires a *profile* - a family of measures parameterized by a sensitivity parameter.

Consider the classic Hill numbers from ecology. Given a species abundance distribution, the Hill number of order q is:

```
qD = (Σ_i p_i^q)^(1/(1-q))
```

Different values of q emphasize different aspects:

| q | Emphasis | Interpretation |
|---|----------|----------------|
| 0 | All species equally | Richness (count of species) |
| 1 | Proportional to frequency | Effective number of "typical" species |
| 2 | Favors abundant species | Effective number of "dominant" species |
| ∞ | Only the most common | Inverse of max probability |

**The key insight**: Two communities may have the same diversity at q=1 but different diversity at q=0 or q=2. Their *profiles* (diversity as a function of q) may cross. When profiles cross, the diversity ordering is **ambiguous** - it depends on whether you care more about rare or common species.

The same principle applies to normativity measurement:

- **Mean-based statistics** (q=1): What is the expected compliance? Treats all trajectories proportionally.
- **Mode-based statistics** (q→∞): What is the compliance of the most probable trajectory? Focuses on dominant outputs.
- **Variance vs. mode-deviation**: Mean deviation from the mean vs. deviation from the mode yield different pictures.

A distribution could have:
- Low variance around the mean but high deviation from the mode (bimodal)
- High variance but most mass near the mode (heavy tails)

**No single statistic captures the full picture.** The generalized core provides a profile that reveals different facets of normativity.

### The (q, r) Parameterization

The core admits a parametrized family with two parameters (q, r):

```
⟨Λ_n⟩_{q,r}
```

- **r (escort order)**: Which trajectories get attention
  - r=1: Actual distribution
  - r=0: Uniform over support
  - r=∞: Mode (most probable)
  - r=-∞: Anti-mode (rarest)

- **q (power mean order)**: How compliance values are aggregated
  - q=1: Arithmetic mean
  - q=0: Geometric mean
  - q=-1: Harmonic mean
  - q=∞: Max
  - q=-∞: Min

### Named Variants

| Name | q | r | Description |
|------|---|---|-------------|
| `standard` | 1 | 1 | Standard expected compliance |
| `uniform` | 1 | 0 | Uniform average over support |
| `mode` | 1 | ∞ | Compliance of most probable |
| `max` | ∞ | 1 | Max compliance in distribution |
| `mode_min` | -∞ | ∞ | Min compliance among modes |
| `confident` | 1 | 2 | Confident core (favors high-prob) |
| `rms` | 2 | 1 | Root-mean-square |
| `antimode` | 1 | -∞ | Compliance of rarest |
| `geometric` | 0 | 1 | Geometric mean |
| `harmonic` | -1 | 1 | Harmonic mean |

## Deviance Statistics

### Expected Deviance

```
E[∂_n] = Σ_i w_i * ||Λ_n(x_i) - ⟨Λ_n⟩||
```

- **Low E[∂]**: Homogenized - outputs cluster around the core
- **High E[∂]**: Diverse - outputs spread away from the core

### Deviance Variance

```
Var[∂_n] = E[∂_n²] - E[∂_n]²
```

Measures how consistent the deviations are.

## Per-Branch Analysis

Estimation is computed separately for each branch group:

```
Group 0 (trunk):    ⟨Λ_n⟩_trunk, E[∂]_trunk, ...
Group 1 (branch_1): ⟨Λ_n⟩_branch_1, E[∂]_branch_1, ...
Group 2 (branch_2): ⟨Λ_n⟩_branch_2, E[∂]_branch_2, ...
```

This reveals how branching affects the normative pattern.

## Output Format

Estimation outputs are saved to `out/est_<method>_<gen>_<scoring>.json`:

```json
{
  "groups": [
    {
      "group_idx": 0,
      "name": "trunk",
      "core": [0.8, 0.6, 0.3],
      "core_inv_ppl": [0.75, 0.55, 0.35],
      "deviance_avg": 0.15,
      "deviance_var": 0.02,
      "deviance_avg_inv_ppl": 0.18,
      "deviance_var_inv_ppl": 0.03,
      "trajectories": [
        {
          "traj_idx": 0,
          "orientation": [0.2, -0.1, 0.05],
          "deviance": 0.23
        }
      ],
      "core_variants": [
        {
          "name": "standard",
          "q": 1.0,
          "r": 1.0,
          "core": [0.8, 0.6, 0.3],
          "deviance_avg": 0.15
        }
      ]
    }
  ],
  "categorical_judgements": [...],
  "structure_info": [
    {
      "label": "c1",
      "description": "Does this mention a person? + Does this mention a boy?",
      "is_grouped": true,
      "questions": [...]
    }
  ],
  "branch_rates": [
    {
      "branch": "trunk",
      "trajectory_count": 10,
      "structure_rates": {"c1": 0.8, "c2": 0.6},
      "question_rates": {"c1": {"Does this mention a person?": 0.9, ...}}
    }
  ]
}
```

## Summary Output

A human-readable summary is saved to `out/summary_est_<...>.json`:

```json
{
  "structures": [
    {"label": "c1", "description": "...", "is_grouped": true}
  ],
  "branch_rates": [...],
  "branch_cores": [
    {
      "branch": "trunk",
      "prob_weighted_core": {"c1": 0.8, "c2": 0.6},
      "inv_ppl_weighted_core": {"c1": 0.75, "c2": 0.55},
      "deviance_avg": 0.15
    }
  ]
}
```

## Interpreting Results

### Core Values

- Values near 1.0: Structure is normatively satisfied (model defaults to "yes")
- Values near 0.0: Structure is normatively violated (model defaults to "no")
- Values near 0.5: Structure is uncertain/balanced

### Comparing Branches

If `core_boy[c1] = 0.9` and `core_cat[c1] = 0.3`:
- Branching on "boy" strongly activates structure c1
- Branching on "cat" suppresses structure c1

### Deviance Interpretation

- **E[∂] ≈ 0**: Perfect homogenization (all outputs match the core)
- **E[∂] high**: High diversity (outputs vary from the core)
- **E[∂] → 0 as model trains**: Potential sign of mode collapse

### Weighting Comparison

- **Prob-weighted**: What the model actually outputs (frequency)
- **Inv-ppl-weighted**: What the model is confident about (quality)

If these differ significantly, the model may produce frequent low-confidence outputs.

## Practical Considerations

### Sample Size

More trajectories give better core estimates. With few samples, the core may be noisy.

### Structure Correlation

Correlated structures (e.g., "mentions man" and "masculine") will have correlated core values. Consider this when interpreting.

### Zero Probabilities

Trajectories with very low probability may have unstable weights. Numerical safeguards prevent underflow.

### Comparing Across Experiments

Core values are relative to the specific prompt, structures, and model. Compare within experiments rather than across.
