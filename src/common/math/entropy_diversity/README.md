# Entropy and Diversity Module

Unified Rényi-Hill framework for entropy, diversity, and divergence calculations with support for native Python, NumPy, and PyTorch. Includes structure-aware metrics for measuring compliance deviations relative to normative cores.

## Module Organization

### Facade Functions (Public API)

- **`entropy_facade.py`** - Rényi entropy H_q and Shannon entropy
- **`diversity_facade.py`** - Hill numbers D_q (diversity) and concentration 1/D_q
- **`divergence_facade.py`** - KL divergence, Rényi divergence, Jensen-Shannon divergence
- **`power_mean.py`** - Generalized means M_α (power means for raw values and probabilities)
- **`escort_distribution.py`** - Escort distributions (q-tilted views of probability masses)
- **`common_orders.py`** - Convenience wrappers for standard parameters (richness, Shannon/Simpson diversity, etc.)
- **`structure_aware.py`** - Structure-aware metrics (compliance, orientation, deviance, cores)

### Implementation Modules

- **`entropy_impl.py`** - Backend for Rényi entropy (native, NumPy, PyTorch)
- **`diversity_impl.py`** - Backend for Hill numbers and concentration (native, NumPy, PyTorch)
- **`divergence_impl.py`** - Backend for KL, Rényi, and Jensen-Shannon divergence (native, NumPy, PyTorch)
- **`power_mean_impl.py`** - Backend for power means (native, NumPy, PyTorch)
- **`escort_distribution_impl.py`** - Backend for escort distributions (native, NumPy, PyTorch)
- **`entropy_primitives.py`** - Low-level operations (probs ↔ logprobs, log-sum-exp, surprise, rarity)
- **`core_impl.py`** - Shared constants and primitives (_EPS, log-sum-exp implementations)

## Key Concepts

### Rényi Entropy and Hill Numbers

Rényi entropy H_q of order q unifies all standard entropy measures:

```
H_q = (1/(1-q)) · log(Σ p_i^q)

Special cases:
  q=0:   log(S)           (Hartley entropy)
  q=1:   -Σ p_i log(p_i)  (Shannon entropy)
  q=2:   -log(Σ p_i²)     (collision entropy)
  q→∞:   -log(max p_i)    (min-entropy)
```

Hill numbers D_q = exp(H_q) express entropy as "effective number of categories":

```
D_q = exp(H_q)

Special cases:
  q=0:   S                (richness)
  q=1:   exp(H)           (Shannon diversity)
  q=2:   1/Σ p_i²         (Simpson diversity)
  q→∞:   1/max p_i        (Berger-Parker index)
```

Concentration 1/D_q measures how concentrated a distribution is.

### Divergence Measures

- **KL divergence** D_KL(p || q) = Σ p_i log(p_i / q_i): asymmetric, unbounded
- **Rényi divergence** D_α(p || q): generalized family including KL at α=1
- **Jensen-Shannon divergence** JSD(p || q): symmetric, bounded in [0, log(2)]

### Power Means

Generalized mean of order α:

```
M_α(x) = (Σ x_i^α / n)^(1/α)

Special cases:
  α→-∞:  min(x)
  α=-1:  harmonic mean
  α→0:   geometric mean
  α=1:   arithmetic mean
  α=2:   quadratic mean (RMS)
  α→+∞:  max(x)
```

For probabilities, power_mean_from_logprobs computes M_α(p) from log-probabilities in a numerically stable way.

### Escort Distributions

The escort distribution of order q shows how probability masses "look" through a q-lens:

```
π_i^(q) = p_i^q / Σ p_j^q

Special cases:
  q→0:   uniform over support (democratic lens)
  q=1:   original distribution (no distortion)
  q=2:   dominant species amplified
  q→∞:   all mass on argmax (autocratic lens)
  q<0:   rare species amplified (contrarian lens)
```

### Structure-Aware Metrics

Quantify diversity relative to context-specific structures:

- **Compliance α_i(x)** ∈ [0,1]: How much string x satisfies structure i
- **System compliance Λ_n(x)**: Vector of compliances across n structures
- **System core ⟨Λ_n⟩**: Expected compliance under a distribution
- **Orientation θ_n(x) = Λ_n(x) - ⟨Λ_n⟩**: Deviation from the normative core
- **Deviance ∂_n(x) = ||θ_n(x)||**: Scalar measure of non-normativity

Deviance measures (KL-based and symmetric):
- **Excess deviance** ∂⁺ = exp(D_α(compliance || core)): over-compliance
- **Deficit deviance** ∂⁻ = exp(D_α(core || compliance)): under-compliance
- **Mutual deviance** ∂_M = exp(JSD(compliance || core)): symmetric measure

## Quick Reference

```python
from src.common.math.entropy_diversity import (
    # Entropy and diversity
    renyi_entropy, shannon_entropy,
    q_diversity, q_concentration,

    # Divergence
    kl_divergence, renyi_divergence, js_divergence,

    # Power means
    power_mean, weighted_power_mean, power_mean_from_logprobs,

    # Escort distribution
    escort_logprobs, escort_probs,

    # Named convenience functions
    richness, shannon_diversity, simpson_diversity,
    shannon_concentration, simpson_concentration,
    geometric_mean_prob, arithmetic_mean_prob, harmonic_mean_prob,
    min_prob, max_prob,

    # Structure-aware metrics
    orientation, deviance, normalized_deviance,
    core_entropy, core_diversity, normalize_core,
    generalized_structure_core, generalized_system_core,
    excess_deviance, deficit_deviance, mutual_deviance,
    expected_deviance, deviance_variance, expected_orientation,
    expected_excess_deviance, expected_deficit_deviance, expected_mutual_deviance,

    # Primitives
    probs_to_logprobs, logprobs_to_probs,
    log_sum_exp, surprise, rarity,
)
```

## Implementation Details

All facade functions support three backends selected automatically:
- **Native Python** (Sequence[float]) - using math library
- **NumPy** (np.ndarray) - using NumPy and scipy.special
- **PyTorch** (torch.Tensor) - using PyTorch operations

Numerical stability is ensured through:
- log-sum-exp trick for entropy computation
- Working in log-space for probability operations
- Clipping values away from exact zero (_EPS = 1e-12)
