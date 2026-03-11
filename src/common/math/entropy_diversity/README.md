# Entropy and Diversity Module

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Core entropy and diversity theory implementing the unified Renyi-Hill framework and structure-aware diversity metrics.

## Contents

### Public API

- `entropy.py` - Renyi entropy H_q, Shannon entropy (q=1)
- `diversity.py` - Hill numbers D_q, concentration 1/D_q
- `divergence.py` - KL divergence, Renyi divergence
- `power_mean.py` - Power mean M_alpha, weighted power mean
- `escort_distribution.py` - Escort distributions (q-tilted view)
- `common_orders.py` - Named wrappers (richness, shannon_diversity, simpson_diversity, etc.)
- `structure_aware.py` - Compliance, orientation, deviance, core statistics

### Implementation Details

- `*_impl.py` files - Backend implementations (native Python, NumPy, PyTorch)
- `entropy_primitives.py` - Low-level operations (probs_to_logprobs, log_sum_exp, surprise, rarity)
- `core_impl.py` - Shared implementation primitives

## Quick Reference

```python
from src.common.math.entropy_diversity import (
    # Entropy/Diversity
    renyi_entropy, shannon_entropy,       # H_q
    q_diversity, q_concentration,         # D_q, 1/D_q
    kl_divergence, renyi_divergence,      # Divergence

    # Power means
    power_mean, weighted_power_mean,
    power_mean_from_logprobs,

    # Escort distribution
    escort_logprobs, escort_probs,

    # Named orders
    richness, shannon_diversity, simpson_diversity,
    shannon_concentration, simpson_concentration,
    geometric_mean_prob, arithmetic_mean_prob,

    # Structure-aware
    orientation, deviance, normalized_deviance,
    core_entropy, core_diversity,
    generalized_structure_core, generalized_system_core,
    expected_deviance, deviance_variance, expected_orientation,
    excess_deviance, deficit_deviance,
)
```

## Key Relationships

```
H_q = (1/(1-q)) log(sum(p_i^q))    Renyi entropy
D_q = exp(H_q)                     Hill number (effective count)
1/D_q = exp(-H_q)                  Concentration

theta = Lambda - <Lambda>          Orientation (deviation from norm)
d = ||theta||                      Deviance (scalar non-normativity)
```

See `../EXPLANATION.md` for detailed documentation.
