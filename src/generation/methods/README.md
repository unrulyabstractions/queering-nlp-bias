# Generation Methods

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Method implementations for trajectory generation.

## Quick Links

- [../EXPLANATION.md](../EXPLANATION.md) - Detailed algorithm explanations
- [../ADDING_METHOD.md](../ADDING_METHOD.md) - How to add new methods

## Files

| File | Description |
|------|-------------|
| `simple_sampling_method.py` | Temperature sampling (N samples per arm) |
| `forking_paths_method.py` | Greedy + fork at high-entropy positions |
| `forking_paths_params.py` | ForkingParams (separate for circular import avoidance) |
| `forking_paths_types.py` | PositionAnalysis, ForkPoint, etc. |
| `entropy_seeking_method.py` | Iterative entropy-guided tree expansion |
| `entropy_seeking_types.py` | TreePath, ExpansionPoint |
| `just_greedy_method.py` | Single greedy trajectory per arm |
| `logging/` | Method-specific logging utilities |

## Method Interface

```python
@dataclass
class MyParams(GenerationMethodParams):
    my_param: int = 10
    name: ClassVar[str] = "my-method"

@register_method(MyParams)
def generate_my_method(runner, config, params, log_fn=None) -> ArmGenerationResult:
    ...
```

## File Organization

Methods with separate logging modules should split params into `*_params.py` to avoid circular imports between method and logging modules.
