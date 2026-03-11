# Math Module

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Mathematical utilities for LLM analysis: probability, entropy, diversity, and trajectory metrics.

## Structure

```
math/
├── entropy_diversity/     # Core theory (entropy, diversity, divergence, power mean)
├── num_types.py           # Type aliases (Num, Nums) with tensor/numpy dispatch
├── math_primitives.py     # Low-level helpers (argmin, argmax, normalize)
├── probability.py         # Log-probability normalization and weighting
├── aggregation_methods.py # Aggregation strategies (mean, median, etc.)
├── trajectory_metrics.py  # Sequence metrics (perplexity, cross-entropy)
├── branch_metrics.py      # Distribution metrics for branches
├── fork_metrics.py        # Binary choice metrics
└── faithfulness_scores.py # Faithfulness scoring utilities
```

## Quick Reference

### Entropy & Diversity (from `entropy_diversity/`)

```python
from src.common.math import (
    renyi_entropy, shannon_entropy,    # Entropy H_q
    q_diversity, q_concentration,      # Hill numbers D_q
    kl_divergence, renyi_divergence,   # Divergence D_alpha
    power_mean, weighted_power_mean,   # Generalized means M_alpha
)
```

### Trajectory Metrics

```python
from src.common.math import perplexity, inv_perplexity, empirical_cross_entropy
```

### Structure-Aware Diversity

```python
from src.common.math import (
    orientation, deviance, normalized_deviance,
    core_entropy, core_diversity,
    generalized_structure_core, generalized_system_core,
    expected_deviance, deviance_variance,
)
```

## Type System

All functions accept `Nums` (sequences/arrays/tensors) and return `Num`:

```python
Num = float | np.floating | torch.Tensor
Nums = Sequence[float] | np.ndarray | torch.Tensor
```

Dispatch is automatic: pass Python lists, NumPy arrays, or PyTorch tensors.

See `EXPLANATION.md` for detailed documentation.
