# Generation Package

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Trajectory generation using various sampling strategies.

## Quick Links

- [EXPLANATION.md](EXPLANATION.md) - Detailed algorithm explanations, data flow, and architecture
- [ADDING_METHOD.md](ADDING_METHOD.md) - Guide for adding new generation methods
- [methods/](methods/) - Method implementations

## Directory Structure

```
generation/
├── generation_config.py           # GenerationConfig with prompt, arms, method params
├── generation_output.py           # GenerationOutput for saving results
├── generation_pipeline.py         # run_generation_pipeline() entry point
├── generation_types.py            # Arm, ForkArm, ArmGenerationResult
├── generation_method_registry.py  # @register_method decorator and registry
├── methods/                       # Method implementations
│   ├── simple_sampling_method.py  # Temperature sampling
│   ├── forking_paths_method.py    # Greedy + fork exploration
│   ├── entropy_seeking_method.py  # Entropy-guided expansion
│   ├── just_greedy_method.py      # Single greedy path
│   └── logging/                   # Method-specific logging
├── ADDING_METHOD.md               # How to add new methods
└── EXPLANATION.md                 # Detailed algorithm documentation
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

| Method | Description |
|--------|-------------|
| `simple-sampling` | Sample N trajectories per arm using temperature |
| `forking-paths` | Greedy path + fork at high-entropy positions |
| `seeking-entropy` | Iteratively expand tree at highest-entropy points |
| `just-greedy` | One greedy (temperature=0) trajectory per arm |

## Adding New Methods

Create one file in `methods/` with a params class and decorated function:

```python
@dataclass
class MyParams(GenerationMethodParams):
    my_param: int = 10
    name: ClassVar[str] = "my-method"

@register_method(MyParams)
def generate_my_method(runner, config, params, log_fn=None):
    ...
```

See [ADDING_METHOD.md](ADDING_METHOD.md) for details.
