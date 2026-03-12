# Generation Package

Trajectory generation using various sampling strategies.

## Quick Links

- [EXPLANATION.md](EXPLANATION.md) - Detailed algorithm explanations and data structures
- [ADDING_METHOD.md](ADDING_METHOD.md) - Guide for adding new generation methods
- [methods/](methods/) - Method implementations

## Directory Structure

```
generation/
├── generation_config.py           # GenerationConfig with prompt, arms, method params
├── generation_output.py           # GenerationOutput for saving results
├── generation_helpers.py          # Helper functions for output formatting
├── generation_pipeline.py         # run_generation_pipeline() entry point
├── generation_method_registry.py  # @register_method decorator and registry
├── methods/                       # Method implementations
│   ├── simple_sampling_method.py  # Simple temperature sampling
│   ├── forking_paths_method.py    # Greedy path + entropy-based forking
│   ├── entropy_seeking_method.py  # Iterative entropy-guided expansion
│   ├── just_greedy_method.py      # Single greedy path per arm
│   └── logging/                   # Method-specific logging utilities
├── ADDING_METHOD.md               # How to add new methods
└── EXPLANATION.md                 # Algorithm specifications
```

## Quick Start

```python
from src.generation import GenerationConfig, run_generation_pipeline
from src.inference import ModelRunner

config = GenerationConfig.load("trials/generation/example.json")
runner = ModelRunner(config.model)

result = run_generation_pipeline(runner, config, method="simple-sampling")
print(f"Generated {len(result.output.tree.trajs)} trajectories")
```

## Available Methods

| Method | Description | Output |
|--------|-------------|--------|
| `simple-sampling` | Sample N trajectories per arm | `out/simple-sampling/<trial>/generation.json` |
| `forking-paths` | Greedy path + exploration at high-entropy points | `out/forking-paths/<trial>/generation.json` |
| `seeking-entropy` | Iteratively expand tree at highest-entropy nodes | `out/seeking-entropy/<trial>/generation.json` |
| `just-greedy` | One greedy trajectory per arm | `out/just-greedy/<trial>/generation.json` |

## Adding New Methods

Create one file in `methods/` with a params class and decorated function:

```python
@dataclass
class MyParams(GenerationMethodParams):
    my_param: int = 10
    name: ClassVar[str] = "my-method"

@register_method(MyParams)
def generate_my_method(runner, config, params, log_fn=None):
    # Return ArmGenerationResult
```

See [ADDING_METHOD.md](ADDING_METHOD.md) for details.
