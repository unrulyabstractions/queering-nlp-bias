# Schemas

Data schemas and configuration classes for the experiment pipeline.

## Contents

### Configuration Schemas
- `generation.py` - `GenerationConfig`, `OutputPaths` for generation stage
- `scoring.py` - `ScoringConfig` for scoring stage
- `estimation.py` - `EstimationOutput`, `GroupEstimate`, `JudgmentData` for estimation
- `default_config.py` - Default parameter values

### Display Utilities
- `experiment.py` - `ExperimentResult`, `GroupResult`, display functions
- `script_utils.py` - Logging and formatting utilities
- `log_utils.py` - Log banner and divider helpers

## Key Classes

```python
from scripts.schemas import (
    GenerationConfig,    # Load/validate generation config
    ScoringConfig,       # Load/validate scoring config
    OutputPaths,         # Paths for pipeline outputs
    EstimationOutput,    # Full estimation results
    JudgmentData,        # Loaded judgment data for estimation
)
```
