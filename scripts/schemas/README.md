# Scripts / Schemas

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Script utilities for the experiment pipeline.

## Contents

| File | Purpose |
|------|---------|
| `script_utils.py` | Argument parsing, model loading, and logging utilities |

## What This Module Provides

### Argument Parsing

`parse_generation_args()` handles CLI argument parsing for generation scripts:
- Loads generation config from JSON
- Applies CLI overrides to config parameters
- Sets random seed for reproducibility

```python
from schemas.script_utils import parse_generation_args, ArgSpec

parsed = parse_generation_args(
    description="My generation script",
    examples=["config.json", "config.json --samples-per-arm 10"],
    extra_args=[
        ArgSpec("samples-per-arm", int, "N", "Trajectories per arm"),
    ],
)

config = parsed.config       # GenerationConfig with CLI overrides applied
config_path = parsed.config_path
```

### Model Loading

`load_model()` loads the model from config and logs model type:

```python
from schemas.script_utils import load_model

runner = load_model(config)  # Returns ModelRunner, logs device and model type
```

### Logging Utilities

```python
from schemas.script_utils import log_prompt_header, log_experiment_start

# Log prompt structure at generation start
log_prompt_header(config.prompt, config.trunk, config.branches)

# Log experiment header with parameters
log_experiment_start("EXPERIMENT", paths, method="simple-sampling")
```

### CLI Override Application

`apply_cli_overrides_to_config()` maps CLI arguments to method parameters:

```python
from schemas.script_utils import apply_cli_overrides_to_config

overrides = {"samples_per_arm": 20, "max_alternates": 5}
apply_cli_overrides_to_config(config, overrides)
```

## Main Schema Classes

The main schema classes are in `src/`:

```python
from src.generation import GenerationConfig, GenerationOutput
from src.scoring import ScoringConfig, ScoringOutput
from src.estimation import ScoringData, EstimationOutput
```

See `src/` for full documentation.
