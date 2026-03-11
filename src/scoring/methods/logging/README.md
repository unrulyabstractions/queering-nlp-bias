# Scoring Method Logging

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Logging utilities for scoring methods.

## Contents

- `scoring_logging.py` - Common logging utilities for scoring

## Usage

Scoring methods accept an optional `log_fn` callback:

```python
def my_log_fn(message: str) -> None:
    print(message)

result = run_scoring_pipeline(config, gen_output, log_fn=my_log_fn)
```
