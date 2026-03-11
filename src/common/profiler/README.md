# Profiler Module

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Simple profiling utilities for timing code execution.

## Contents

- `timer.py` - Hierarchical profiler with context manager API
- `decorators.py` - Function decorators for profiling

## Usage

The singleton profiler `P` provides a simple API for timing:

```python
from src.common.profiler import P

# Context manager (recommended)
with P("load_data"):
    data = load()

# Manual start/stop
P.start("train")
# ... training ...
P.stop("train")

# Nested timing (builds hierarchy)
with P("outer"):
    with P("inner"):
        work()

# Report summary
P.report()      # Print timing report
P.summary()     # Get dict of name -> ms
P.reset()       # Clear all timings

# Enable/disable
P.disable()     # No-op mode
P.enable()      # Resume profiling
```
