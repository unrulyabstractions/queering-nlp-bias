# Scripts / Schemas

Shared utilities for pipeline scripts across generation, scoring, estimation, and experiment workflows.

## Module Contents

| File | Purpose |
|------|---------|
| `script_utils.py` | Argument parsing, model loading, logging, and pipeline orchestration |

## Utilities

### Argument Parsing

`parse_generation_args()` provides unified CLI argument parsing for generation scripts:

```python
from scripts.schemas.script_utils import parse_generation_args, ArgSpec

parsed = parse_generation_args(
    description="Generate trajectories",
    examples=["config.json", "config.json --samples-per-arm 10"],
    extra_args=[ArgSpec("samples-per-arm", int, "N", "Trajectories per arm")],
)
config = parsed.config
config_path = parsed.config_path
```

- Loads generation config from JSON
- Applies CLI overrides to method parameters
- Sets random seed from config

### Model Loading

`load_model()` initializes the model runner and logs model type (base vs. chat/instruct):

```python
from scripts.schemas.script_utils import load_model

runner = load_model(config)
```

### Logging Utilities

Log generation setup and experiment metadata:

```python
from scripts.schemas.script_utils import log_prompt_header, log_experiment_start, log_output_paths

log_prompt_header(config.prompt, config.trunk, config.branches, config.twig_variations)
log_experiment_start("EXPERIMENT", paths, method="simple-sampling")
log_output_paths(paths)
```

### CLI Override Application

Map CLI arguments to method parameters:

```python
from scripts.schemas.script_utils import apply_cli_overrides_to_config

apply_cli_overrides_to_config(config, {"samples_per_arm": 20})
```

### Generation Pipeline Runner

Execute the full generation workflow in a single call:

```python
from scripts.schemas.script_utils import run_generation_script

output = run_generation_script(
    config=config,
    config_path=config_path,
    method="simple-sampling",
    section_title="Simple Sampling",
)
```

Handles: model loading, logging, pipeline execution, trajectory logging, file saving, and summary display.
