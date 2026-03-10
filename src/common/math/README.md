# Math Module

Mathematical utilities for LLM analysis, including probability calculations, metrics, and aggregation methods.

## Contents

- `entropy_diversity/` - Entropy, diversity, and structure-aware metrics (see submodule)
- `math_primitives.py` - Basic mathematical operations
- `num_types.py` - Numeric type definitions (supports both Python and PyTorch)
- `aggregation_methods.py` - Methods for aggregating values (means, etc.)
- `trajectory_metrics.py` - Trajectory-level mathematical metrics
- `branch_metrics.py` - Branch-level metrics
- `fork_metrics.py` - Fork point metrics
- `faithfulness_scores.py` - Faithfulness scoring utilities

## Usage

```python
from src.common.math import weighted_power_mean, shannon_entropy
from src.common.math.entropy_diversity import expected_deviance, generalized_system_core
```
