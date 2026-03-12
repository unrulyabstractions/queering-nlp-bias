# Dynamics Logging

Logging utilities for dynamics computation. Provides structured logging callbacks to display progress during measurements.

## Functions

- **`log_dynamics_header()`** - Log header with trajectory count, arm count, and min points
- **`log_trajectory_start()`** - Log start of trajectory dynamics computation with metadata
- **`log_point_computation()`** - Log measurement at a token position with deviances to all arms
- **`log_trajectory_summary()`** - Log summary for a trajectory
- **`log_dynamics_result()`** - Log final summary with per-arm breakdown

## Usage

Dynamics computation accepts an optional `log_fn` callback (type `LogFn`):

```python
from src.estimation.dynamics.logging import (
    log_dynamics_header,
    log_point_computation,
)

def my_log_fn(message: str) -> None:
    print(message)

# Log dynamics computation header
log_dynamics_header(
    n_trajectories=10,
    n_arms=4,
    min_points=8,
    log_fn=my_log_fn,
)

# Log measurement at a position
log_point_computation(
    position=50,
    n_tokens=200,
    deviances={"root": 0.1, "trunk": 0.2, "branch_1": 0.15},
    own_arm="branch_1",
    log_fn=my_log_fn,
)
```

## Output Format

Example output during dynamics computation:

```
==0== Dynamics Computation
  ──────────────────────────────────────────────────
  Trajectories to analyze: 4
  Arms: 4
  Min measurement points per trajectory: 8
────────────────────────────────────────────────────────────

  [0] root: 156 tokens
      Text: "The protagonist walked slowly through the..."
      @  20 ( 12.8%): root=0.023*, trunk=0.156, branch_1=0.089
      @  40 ( 25.6%): root=0.041*, trunk=0.178, branch_1=0.102
      ...
      Summary: 8 points, final own-arm deviance: 0.0521
```
