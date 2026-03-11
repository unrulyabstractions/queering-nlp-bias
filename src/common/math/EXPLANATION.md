# Mathematical Utilities

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This module provides a unified framework for probability, entropy, diversity, and related metrics used in LLM analysis.

## Conceptual Overview

### The Unified Diversity Framework

The entropy_diversity subpackage implements a unified theory where entropy, diversity, and concentration are all views of the same underlying mathematics:

```
Entropy H_q     <---->     Diversity D_q     <---->     Concentration 1/D_q
   (nats)              (effective count)              (inverse count)

        D_q = exp(H_q)          1/D_q = exp(-H_q)
```

All are parameterized by order `q`:
- **q = 0**: Richness (count of non-zero categories)
- **q = 1**: Shannon (balanced sensitivity to common and rare)
- **q = 2**: Simpson (emphasizes dominant categories)
- **q -> infinity**: Only considers the mode

### Key Distinction: Distributions vs Sequences

The module distinguishes between:

1. **Distributions** (entropy_diversity/): A probability vector summing to 1
   - Input: `logprobs` where `sum(exp(logprobs)) = 1`
   - Metrics: entropy, diversity, divergence

2. **Sequences** (trajectory_metrics.py): A series of conditional probabilities
   - Input: `logprobs` where each is `log P(token_i | context)`
   - Metrics: perplexity, cross-entropy

## Core Functions

### Entropy (`entropy.py`)

```python
def renyi_entropy(logprobs: Nums, q: float) -> Num:
    """Renyi entropy of order q: H_q = (1/(1-q)) * log(sum(p_i^q))"""

def shannon_entropy(logprobs: Nums) -> Num:
    """Shannon entropy (q=1): H = -sum(p_i * log(p_i))"""
```

Special cases of Renyi entropy:
- H_0 = log(richness) - Hartley entropy
- H_1 = Shannon entropy (via L'Hopital)
- H_2 = -log(sum(p_i^2)) - Collision entropy
- H_inf = -log(max(p_i)) - Min-entropy

### Diversity (`diversity.py`)

```python
def q_diversity(logprobs: Nums, q: float) -> Num:
    """Hill number D_q = exp(H_q): effective number of categories"""

def q_concentration(logprobs: Nums, q: float) -> Num:
    """Concentration 1/D_q: how peaked is the distribution"""
```

Hill numbers unify common diversity indices:
- D_0 = richness (count of categories)
- D_1 = exp(Shannon entropy)
- D_2 = 1/sum(p_i^2) = Simpson diversity
- D_inf = 1/max(p_i) = Berger-Parker index

### Divergence (`divergence.py`)

```python
def kl_divergence(p: Nums, q: Nums, normalize: bool = True) -> Num:
    """KL divergence D_KL(p || q) = sum(p_i * log(p_i / q_i))"""

def renyi_divergence(p: Nums, q: Nums, alpha: float = 1.0) -> Num:
    """Renyi divergence of order alpha, generalizing KL"""
```

Divergence measures "distance" between distributions (asymmetric, not a metric).

### Power Mean (`power_mean.py`)

```python
def power_mean(values: Nums, alpha: float) -> Num:
    """Generalized mean M_alpha = (mean(x^alpha))^(1/alpha)"""

def weighted_power_mean(values: Nums, weights: Nums, alpha: float) -> Num:
    """Weighted power mean with probability weights"""

def power_mean_from_logprobs(logprobs: Nums, alpha: float) -> Num:
    """Power mean of probabilities, computed stably from logprobs"""
```

Power mean hierarchy:
- alpha -> -inf: minimum
- alpha = -1: harmonic mean
- alpha -> 0: geometric mean
- alpha = 1: arithmetic mean
- alpha -> +inf: maximum

### Escort Distribution (`escort_distribution.py`)

```python
def escort_logprobs(logprobs: Nums, q: float) -> Nums:
    """Q-tilted view: pi_i^(q) = p_i^q / sum(p_j^q)"""
```

The escort distribution shows how a distribution "looks" at different orders:
- q -> 0: uniform over support (democratic)
- q = 1: original distribution
- q > 1: amplifies dominant categories
- q -> inf: all mass on argmax (autocratic)

### Common Orders (`common_orders.py`)

Convenience wrappers for frequently-used parameter values:

```python
# Diversity
richness(logprobs)           # D_0
shannon_diversity(logprobs)  # D_1
simpson_diversity(logprobs)  # D_2

# Concentration
shannon_concentration(logprobs)  # 1/D_1
simpson_concentration(logprobs)  # 1/D_2

# Power mean of probabilities
geometric_mean_prob(logprobs)   # M_0 = 1/perplexity
arithmetic_mean_prob(logprobs)  # M_1
harmonic_mean_prob(logprobs)    # M_{-1}
min_prob(logprobs)              # M_{-inf}
max_prob(logprobs)              # M_{+inf}
```

## Structure-Aware Diversity (`structure_aware.py`)

Implements the "Queering NLP Bias" framework for measuring diversity relative to normative structures.

### Core Concepts

**Structure**: A property of interest (e.g., "mentions women", "uses formal language")

**StructureCompliance** alpha_i(x): How much string x satisfies structure i (in [0,1])

**SystemCompliance** Lambda_n(x): Vector of compliances across n structures

**SystemCore** <Lambda_n>: Expected compliance under the distribution

**Orientation** theta_n(x): Deviation from core, Lambda_n(x) - <Lambda_n>

**Deviance** d_n(x): Scalar measure of non-normativity, ||theta_n(x)||

### Functions

```python
def orientation(compliance: SystemCompliance, core: SystemCore) -> Nums:
    """theta_n(x) = Lambda_n(x) - <Lambda_n>"""

def deviance(compliance, core, norm: str = "l2") -> Num:
    """Scalar non-normativity: ||theta_n(x)||"""

def normalized_deviance(compliance, core, norm: str = "l2") -> Num:
    """Deviance scaled to [0, 1]"""

def core_entropy(core: SystemCore) -> Num:
    """Entropy of normalized core: how balanced is compliance?"""

def generalized_structure_core(compliances, probs, q=1.0, r=1.0) -> Num:
    """Core with escort weighting (r) and power mean aggregation (q)"""

def expected_deviance(compliances, core, weights=None, norm="l2") -> float:
    """E[d_n]: mean deviance across samples"""

def deviance_variance(compliances, core, weights=None, norm="l2") -> float:
    """Var[d_n]: variance of deviance"""
```

### Relative Entropy Deviance

```python
def excess_deviance(compliance, core, alpha=1.0) -> float:
    """Over-compliance: exp(D_alpha(Lambda || Core))"""

def deficit_deviance(compliance, core, alpha=1.0) -> float:
    """Under-compliance: exp(D_alpha(Core || Lambda))"""
```

## Trajectory Metrics (`trajectory_metrics.py`)

For analyzing sequences of token predictions (not distributions).

```python
def perplexity(logprobs: Sequence[float]) -> float:
    """Effective vocabulary size per token: exp(-mean(logprobs))"""

def inv_perplexity(logprobs: Sequence[float]) -> float:
    """Geometric mean probability: exp(mean(logprobs))"""

def empirical_cross_entropy(logprobs: Sequence[float]) -> float:
    """Average surprise: -mean(logprobs)"""

def alpha_perplexity(logprobs: Sequence[float], alpha: float) -> float:
    """Generalized perplexity using power mean of order alpha"""
```

## Probability Utilities (`probability.py`)

```python
def normalize_log_probs(log_probs: Sequence[float]) -> list[float]:
    """Convert log probs to normalized probabilities (logsumexp trick)"""

def normalize_indexed_log_probs(indexed_log_probs, descending=True):
    """Normalize (index, logprob) pairs with optional sorting"""

def compute_inv_perplexity_weights(log_probs, n_tokens) -> list[float]:
    """Weight sequences by inverse perplexity (per-token confidence)"""
```

## Type System (`num_types.py`)

All functions accept multiple numeric types and dispatch to optimized implementations:

```python
Num = float | np.floating | torch.Tensor  # Scalar
Nums = Sequence[float] | np.ndarray | torch.Tensor  # Array

# Type guards for dispatch
is_tensor(x) -> bool
is_numpy(x) -> bool
```

Usage:
```python
# All work identically:
shannon_entropy([0.5, 0.3, 0.2])           # Python list
shannon_entropy(np.array([0.5, 0.3, 0.2])) # NumPy
shannon_entropy(torch.tensor([0.5, 0.3, 0.2]))  # PyTorch
```

## Numerical Stability

- All functions work with log-probabilities when possible
- Logsumexp trick used throughout
- `_EPS = 1e-12` guards against log(0) and division by zero
- Special handling for -inf (zero probability) cases
