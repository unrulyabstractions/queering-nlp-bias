# Generation Methods

Trajectory generation method implementations. Each method is auto-discovered and registered via the `@register_method` decorator.

## Quick Links

- [../EXPLANATION.md](../EXPLANATION.md) - Detailed algorithm explanations
- [../ADDING_METHOD.md](../ADDING_METHOD.md) - How to add new methods

## Available Methods

| Method | Module | Algorithm |
|--------|--------|-----------|
| `simple-sampling` | `simple_sampling_method.py` | Temperature sampling: N samples per arm |
| `forking-paths` | `forking_paths_method.py` | Greedy path + fork at high-entropy positions with sample continuations |
| `seeking-entropy` | `entropy_seeking_method.py` | Iterative tree expansion at highest-entropy points |
| `just-greedy` | `just_greedy_method.py` | Single greedy trajectory per arm (temperature=0) |

## Method Registry

The registry (`generation_method_registry.py`) provides a unified interface for method discovery:

```python
from src.generation.generation_method_registry import (
    get_method,            # Get the generation function
    get_default_params,    # Get default parameters
    list_methods,          # List all registered method names
    get_output_name,       # Get output directory name (same as method name)
)
```

**Output Paths**: Results are saved to `out/<method>/gen_<trial>.json` and `out/<method>/summary_gen_<trial>.txt`

## Method Interface

To add a new method, implement one file with:

```python
from dataclasses import dataclass
from typing import ClassVar
from src.generation.generation_method_registry import (
    GenerationMethodParams, register_method
)

@dataclass
class MyParams(GenerationMethodParams):
    my_param: int = 10
    name: ClassVar[str] = "my-method"
    _cli_args: ClassVar[dict[str, str]] = {
        "my_param": "--my-param",
    }

@register_method(MyParams)
def generate_my_method(
    runner: ModelRunner,
    config: GenerationConfig,
    params: MyParams,
    log_fn: LogFn | None = None,
) -> ArmGenerationResult:
    # Implementation here
    ...
```

The method is immediately available via `get_method("my-method")` and CLI via `--method my-method`.

## File Organization

### Supporting Files

| File | Purpose |
|------|---------|
| `generation_method_utils.py` | Shared utilities (e.g., `compute_arm_token_lengths`) |
| `forking_paths_params.py` | ForkingParams (separate to avoid circular imports) |
| `forking_paths_types.py` | PositionAnalysis, ForkPoint, QualifyingFork, TopKCandidate |
| `entropy_seeking_types.py` | TreePath, ExpansionPoint |
| `logging/` | Method-specific logging utilities |

Methods with logging modules should split parameter classes into separate `*_params.py` files to avoid circular imports between method and logging submodules.
