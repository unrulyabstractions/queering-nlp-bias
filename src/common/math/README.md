# Math Module

Mathematical utilities for LLM analysis: entropy, diversity, probability, and trajectory metrics.

## Structure

```
math/
├── entropy_diversity/          # Core theory (entropy, diversity, divergence, power mean, etc.)
├── num_types.py                # Type aliases (Num, Nums) with auto-dispatch
├── math_primitives.py          # Low-level helpers (argmin, argmax, normalize)
├── probability_utils.py        # Log-probability normalization and weighting
├── aggregation_methods.py      # Aggregation strategies (mean, median, sum, min, max)
├── trajectory_metrics.py       # Sequence metrics (perplexity, cross-entropy, ranks)
├── branch_metrics.py           # Distribution metrics (diversity, entropy, concentration)
├── fork_metrics.py             # Binary choice metrics (diversity, margin, odds)
├── vector_utils.py             # Vector operations (orientation, L2 norm)
└── faithfulness_scores.py      # Intervention faithfulness scores (sufficiency, etc.)
```

## Quick Reference

### Core Entropy & Diversity

```python
from src.common.math import (
    renyi_entropy, shannon_entropy,    # Rényi entropy H_q
    q_diversity, q_concentration,      # Hill numbers D_q and 1/D_q
    kl_divergence, renyi_divergence,   # Divergence D_α(p||q)
    power_mean,                         # Generalized mean M_α
)
```

### Trajectory Metrics

```python
from src.common.math import (
    perplexity, inv_perplexity,        # Effective vocab size, geometric mean prob
    alpha_perplexity, alpha_inv_perplexity,  # Generalized orders
    empirical_cross_entropy,            # Average surprise per token
    surprise_trajectory, rarity_trajectory,  # Per-token metrics
    worst_token_logprob, best_token_logprob,
    token_ranks_from_logits,            # Rank of chosen tokens
)
```

### Branch & Fork Metrics

```python
from src.common.math import (
    q_branch_diversity, q_branch_entropy, q_branch_concentration,
    q_fork_diversity, q_fork_concentration,
    margin, log_odds, winning_prob,
)
```

### Aggregation & Utilities

```python
from src.common.math import (
    AggregationMethod, aggregate,
    normalize_log_probs, compute_inv_perplexity_weights,
)
```

## Type System

All functions accept multiple numeric types with automatic dispatch:

```python
Num = float | np.floating | torch.Tensor          # Scalar
Nums = Sequence[float] | np.ndarray | torch.Tensor  # Array-like
```

Pass Python lists, NumPy arrays, or PyTorch tensors interchangeably.

See `EXPLANATION.md` for detailed documentation.
