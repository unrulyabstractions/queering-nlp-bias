# Scoring Method Logging

Logging utilities for scoring methods. Provides structured logging callbacks to display progress and errors during scoring.

## Functions

- **`log_trajectory_header()`** - Log trajectory metadata with arm name and selected text preview
- **`log_scoring_section()`** - Log section headers for different scoring methods
- **`log_parse_failure()`** - Log warnings when LLM-based method fails to parse a score

## Usage

Scoring methods accept an optional `log_fn` callback (type `LogFn`):

```python
from src.scoring.methods.logging.scoring_logging_utils import log_trajectory_header, log_parse_failure

def my_log_fn(message: str) -> None:
    print(message)

# Log trajectory header
log_trajectory_header(traj=trajectory, idx=0, total=100, log_fn=my_log_fn)

# Log parse failures
log_parse_failure(
    method_name="CATEGORICAL",
    question="Does this text mention a person?",
    raw_response="The model said something weird",
    log_fn=my_log_fn
)
```
