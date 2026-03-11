# Generation Method Logging

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Logging utilities for generation methods.

## Contents

- `common_logging.py` - Shared logging utilities for all methods
- `forking_paths_logging.py` - Logging for forking paths method
- `entropy_seeking_logging.py` - Logging for entropy-seeking method

## Usage

Each method can optionally accept a `log_fn` callback for progress reporting:

```python
def my_log_fn(message: str) -> None:
    print(message)

result = run_generation_pipeline(runner, config, method="forking", log_fn=my_log_fn)
```
