# Profiler Module

Simple profiling utilities for timing code execution.

## Contents

- `profiling_timer.py` - Singleton profiler `P` with hierarchical timing and context manager API
- `profiling_decorators.py` - Function decorator for automatic profiling and memory logging

## Core API

### Singleton Profiler `P`

The `P` object provides timing with automatic parent-child hierarchy tracking:

```python
from src.common.profiler import P

# Context manager (recommended)
with P("load_data"):
    data = load()

# Manual start/stop
P.start("train")
# ... training ...
elapsed = P.stop("train")  # Returns elapsed time in seconds

# Nested timing (builds hierarchy)
with P("outer"):
    with P("inner"):
        work()

# Report and query
P.report(min_ms=0.1)    # Print hierarchical timing report
P.summary()             # Dict of name -> total_ms
P.get("train")          # Get total ms for specific entry
P.reset()               # Clear all timings

# Control
P.disable()             # No-op mode
P.enable()              # Resume profiling
```

### Profiling Decorator

Automatically times function execution and logs memory usage:

```python
from src.common.profiler import profile

@profile
def my_func():
    pass

@profile("custom_name", verbose=True)
def another_func():
    pass
```

The decorator creates a timing entry in `P` and logs memory stats via `log_memory()`.
