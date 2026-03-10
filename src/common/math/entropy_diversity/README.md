# Entropy and Diversity Module

Core entropy and diversity theory implementing the unified framework from Tom Leinster's work and the structure-aware diversity metrics.

## Concepts

- **Entropy**: Shannon entropy, Rényi entropy, and generalizations
- **Diversity**: Hill numbers, effective number of types
- **Escort distributions**: Probability reweighting for generalized means
- **Power means**: Weighted power means with arbitrary exponents
- **Structure-aware metrics**: Compliance, core, orientation, deviance

## Contents

### Core Theory
- `entropy.py` / `entropy_impl.py` - Entropy calculations
- `diversity.py` / `diversity_impl.py` - Diversity measures (Hill numbers)
- `divergence.py` / `divergence_impl.py` - Rényi divergence
- `power_mean.py` / `power_mean_impl.py` - Weighted power means
- `escort_distribution.py` / `escort_distribution_impl.py` - Escort probabilities

### Structure-Aware Metrics
- `structure_aware.py` - Core, orientation, deviance, expected_orientation
- `core_impl.py` - System core implementations

### Utilities
- `entropy_primitives.py` - Low-level entropy operations
- `common_orders.py` - Named parameter values (q=0, 1, 2, ∞)

## Key Functions

```python
from src.common.math.entropy_diversity import (
    # Entropy/Diversity
    shannon_entropy,
    renyi_entropy,
    hill_number,

    # Structure-aware
    orientation,           # θ(x) = Λ(x) - ⟨Λ⟩
    deviance,             # ∂(x) = ||θ(x)||
    expected_deviance,    # E[∂]
    expected_orientation, # E[θ]
    generalized_system_core,  # ⟨Λ⟩_{q,r}

    # Power means
    weighted_power_mean,
    escort_probs,
)
```

## Reference

Based on: https://www.unrulyabstractions.com/pdfs/diversity.pdf
