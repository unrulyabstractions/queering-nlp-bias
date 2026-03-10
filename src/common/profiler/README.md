# Profiler Module

Simple profiling utilities for timing code execution.

## Contents

- `timer.py` - Timer class for measuring execution time
- `decorators.py` - `@profile` decorator for function profiling

## Usage

```python
from src.common.profiler import profile, Timer

@profile
def my_function():
    ...

with Timer("operation"):
    ...
```
