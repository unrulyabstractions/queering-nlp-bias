# Mathematical Utilities

This module provides unified frameworks for entropy, diversity, probability, and metrics for LLM analysis. It bridges abstract mathematical concepts (information theory) with practical LLM metrics (perplexity, trajectory confidence).

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

## Branch Metrics (`branch_metrics.py`)

"Branch" = probability distribution over alternatives at a single position (e.g., next-token distribution).

Input: probabilities p = [p₁, …, pₙ] with Σpᵢ = 1 (or logits).

### Generalized Branch Metrics (order q)

```python
def q_branch_diversity(probs: Nums, q: float) -> Num:
    """Effective number of alternatives at this branch (Hill number D_q).

    q=0: count all non-zero options (richness)
    q=1: Shannon diversity exp(H)
    q=2: Simpson diversity 1/Σpᵢ²
    q→∞: 1/max(p) (dominated by most likely)

    Range: [1, n]. Higher = more choices available.
    """

def q_branch_entropy(probs: Nums, q: float) -> Num:
    """Rényi entropy at this branch (H_q).

    Range: [0, log n]. Lower = more certain.
    """

def q_branch_concentration(probs: Nums, q: float) -> Num:
    """How concentrated is this branch? (1/D_q).

    Range: [1/n, 1]. Higher = more concentrated on few options.
    """

def vocab_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of vocabulary distribution from raw logits.

    Applies log_softmax then shannon_entropy. Range: [0, log |V|].
    """
```

## Fork Metrics (`fork_metrics.py`)

"Fork" = binary decision between two alternatives (A vs B).

Input: raw scores (p_A, p_B), normalized internally.

### Generalized Fork Metrics (order q)

```python
def q_fork_diversity(prob_a, prob_b, q) -> float:
    """Effective number of options at this fork (D_q for binary).

    Range: [1, 2]. 1.0 = one dominates, 2.0 = perfectly balanced.
    """

def q_fork_concentration(prob_a, prob_b, q) -> float:
    """Concentration at this fork (1/D_q).

    Range: [0.5, 1]. Higher = more decisive.
    """

def q_fork_entropy(prob_a, prob_b, q) -> float:
    """Rényi entropy at this fork (H_q).

    Range: [0, ln 2]. Lower = more decisive.
    """
```

### Decision Strength Metrics

```python
def probability_ratio(prob_a, prob_b) -> float:
    """p_A / p_B. > 1 means A wins. ∞ if p_B = 0."""

def log_odds(prob_a, prob_b) -> float:
    """log(p_A / p_B). > 0 means A wins. ±∞ at boundaries."""

def margin(prob_a, prob_b) -> float:
    """Probability margin p_A - p_B (on normalized probs).

    Range: [-1, 1]. > 0 means A wins.
    """

def abs_margin(prob_a, prob_b) -> float:
    """Absolute margin |p_A - p_B|. Range: [0, 1]. Higher = more decisive."""

def winner(prob_a, prob_b) -> int:
    """Which option wins? 0 = A, 1 = B."""

def winning_prob(prob_a, prob_b) -> float:
    """Probability of the winning option. Range: [0.5, 1]. Higher = more decisive."""
```

## Trajectory Metrics (`trajectory_metrics.py`)

For analyzing sequences of token predictions. Input: logprobs = [ℓ₁, …, ℓₘ] where ℓᵢ = log P(wᵢ | w<ᵢ).

### Per-Token Metrics

```python
def surprise_trajectory(logprobs: Sequence[float]) -> list[float]:
    """Surprise at every position: -ℓᵢ"""

def rarity_trajectory(logprobs: Sequence[float]) -> list[float]:
    """Rarity (inverse prob) at every position: exp(-ℓᵢ)"""
```

### Generalized Perplexity (order α)

```python
def alpha_inv_perplexity(logprobs, alpha) -> float:
    """Generalized inverse-perplexity M_α(p) where p = exp(logprobs).

    α → −∞: worst-case token (min pᵢ)
    α = -1: harmonic mean (pessimistic)
    α = 0: geometric mean (standard)
    α = 1: arithmetic mean (optimistic)
    α → +∞: best-case token (max pᵢ)
    """

def alpha_perplexity(logprobs, alpha) -> float:
    """Generalized perplexity PP_α = 1 / M_α(p). Range: [1, ∞). Lower is better."""
```

### Standard Metrics (α=0)

```python
def inv_perplexity(logprobs: Sequence[float]) -> float:
    """Geometric mean token probability: exp(mean(logprobs)). Range: (0, 1]. Higher is better."""

def perplexity(logprobs: Sequence[float]) -> float:
    """Effective vocabulary size: exp(-mean(logprobs)). Range: [1, ∞). Lower is better."""

def empirical_cross_entropy(logprobs: Sequence[float]) -> float:
    """Average surprise per token: -mean(logprobs). Range: [0, ∞). Lower is better."""
```

### Log-Probability Aggregation

```python
def total_logprob(logprobs: Sequence[float]) -> float:
    """Sum of all token log-probs. Range: (−∞, 0]. Length-dependent."""

def partial_logprob(logprobs, start, end) -> float:
    """Sum of logprobs in range [start, end)"""

def worst_token_logprob(logprobs) -> float:
    """Hardest token: min(ℓᵢ). Range: (−∞, 0]."""

def best_token_logprob(logprobs) -> float:
    """Easiest token: max(ℓᵢ). Range: (−∞, 0]."""

def worst_token_position(logprobs) -> int:
    """Index of hardest token: argmin(ℓᵢ)"""

def best_token_position(logprobs) -> int:
    """Index of easiest token: argmax(ℓᵢ)"""
```

### Rank-Based Metrics (require full logits)

```python
def token_ranks_from_logits(token_ids, full_logits) -> list[int]:
    """Rank of each chosen token in vocabulary. Rank 1 = greedy (highest prob)."""

def worst_token_rank(ranks: Sequence[int]) -> int:
    """Worst-ranked token position. Higher = more surprising choice."""

def worst_rank_position(ranks: Sequence[int]) -> int:
    """Position of worst-ranked token"""

def top_p_normalized_logprobs(token_ids, full_logits, p=100) -> list[float]:
    """Logprobs renormalized to top-p tokens only.

    Gives higher probabilities by restricting to plausible alternatives.
    """
```

## Probability Utilities (`probability_utils.py`)

```python
def normalize_log_probs(log_probs: Sequence[float]) -> list[float]:
    """Convert log probabilities to normalized probabilities (logsumexp trick).

    Args: Sequence of log probs
    Returns: List of normalized probs summing to 1.0
    """

def normalize_indexed_log_probs(indexed_log_probs, descending=True) -> list[tuple[int, float]]:
    """Normalize (index, log_prob) pairs with optional sorting.

    Useful for weighted sampling and ranking trajectories.
    """

def compute_inv_perplexity_weights(log_probs, n_tokens) -> list[float]:
    """Weight sequences by per-token confidence (inverse perplexity).

    inv_ppl = exp(log_prob / n_tokens)
    Normalizes sequence quality by token count, not raw probability.
    """

def get_conditional_log_probs(items, condition_name, exclude_branch_mismatch=True) -> list[tuple[int, float]]:
    """Extract conditional log probabilities for a specific condition.

    Extracts (traj_idx, log_prob) pairs from items with conditional_logprobs dict.
    """
```

## Aggregation Methods (`aggregation_methods.py`)

```python
class AggregationMethod(Enum):
    """Enumeration of aggregation strategies."""
    MEAN, MAX, MIN, SUM, MEDIAN

def aggregate(values: Sequence[float], method: AggregationMethod) -> float:
    """Aggregate a sequence of values using the specified method.

    Returns -inf for empty sequences.
    """
```

## Math Primitives (`math_primitives.py`)

Low-level utilities for stability and convenience:

```python
def argmin(xs: Sequence[float]) -> int:
    """Index of minimum value (or 0 for empty list)"""

def argmax(xs: Sequence[float]) -> int:
    """Index of maximum value (or 0 for empty list)"""

def logprob_to_prob(logprob: float) -> float:
    """Convert single log-probability to probability"""

def prob_to_logprob(prob: float) -> float:
    """Convert single probability to log-probability (returns -inf if prob < eps)"""

def normalize(values: Sequence[float]) -> list[float]:
    """Normalize non-negative values to sum to 1.0 using log-sum-exp trick"""

def normalize_pair(a: float, b: float) -> tuple[float, float]:
    """Normalize two values to sum to 1.0 using log-sum-exp"""
```

## Vector Utilities (`vector_utils.py`)

```python
def compute_orientation_vector(source_core, reference_core) -> tuple[list[float], float]:
    """Compute orientation vector and its L2 norm.

    Orientation = source_core - reference_core (element-wise difference).
    Returns (vector, norm). If reference_core is None, returns ([], 0.0).
    """
```

## Faithfulness Scores (`faithfulness_scores.py`)

Intervention analysis utilities for circuit analysis:

```python
# Compute raw metrics
def compute_recovery(y_intervened, y_clean, y_corrupted) -> float:
    """R = (y_intervened - y_corrupted) / (y_clean - y_corrupted)"""

def compute_disruption(y_intervened, y_clean, y_corrupted) -> float:
    """D = (y_clean - y_intervened) / (y_clean - y_corrupted)"""

# 2x2 faithfulness matrix
def compute_sufficiency_score(...) -> float:
    """Denoise IN-circuit: how much does patching recover behavior?"""

def compute_completeness_score(...) -> float:
    """Denoise OUT-circuit: is the circuit complete without others?"""

def compute_necessity_score(...) -> float:
    """Noise IN-circuit: does corrupting it break behavior?"""

def compute_independence_score(...) -> float:
    """Noise OUT-circuit: is the circuit independent of others?"""

# Convenience from recovery
def sufficiency_from_recovery(recovery) -> float  # = recovery
def completeness_from_recovery(recovery) -> float  # = 1 - recovery
def necessity_from_recovery(recovery) -> float  # = 1 - recovery
def independence_from_recovery(recovery) -> float  # = recovery
```

## Type System (`num_types.py`)

All functions accept multiple numeric types with automatic dispatch:

```python
Num = float | np.floating | torch.Tensor  # Scalar
Nums = Sequence[float] | np.ndarray | torch.Tensor  # Array

def is_tensor(x) -> bool:
    """Check if x is a torch.Tensor"""

def is_numpy(x) -> bool:
    """Check if x is a numpy array or numpy scalar"""
```

Usage:
```python
# All work identically:
shannon_entropy([0.5, 0.3, 0.2])           # Python list
shannon_entropy(np.array([0.5, 0.3, 0.2])) # NumPy array
shannon_entropy(torch.tensor([0.5, 0.3, 0.2]))  # PyTorch tensor
```

## Numerical Stability

- All functions work with log-probabilities to avoid underflow
- Logsumexp trick used for stable normalization
- `_EPS = 1e-12` guards against log(0) and division by zero
- Special handling for -inf (zero probability) and inf cases
