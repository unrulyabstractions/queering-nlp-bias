# CLAUDE.md - src/dynamics/

This module analyzes how trajectories evolve by scoring partial text at each token position.

## Configuration

All dynamics parameters are defined in `src/common/default_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DYNAMICS_STEP` | 4 | Measure scores every N tokens |
| `DYNAMICS_TRAJS_PER_ARM` | 2 | Number of most extremal trajectories to analyze per arm |
| `DYNAMICS_ARMS` | `["branch"]` | Arm types to analyze: `"root"`, `"trunk"`, `"branch"`, `"twig"` |

## Main Entry Points

```python
from src.dynamics import compute_dynamics, plot_dynamics, save_dynamics_json
from src.scoring import Scorer

# Build trajectory tuples (with string selection applied)
trajectories = [
    (traj.traj_idx, traj.arm, traj.text, traj.n_generated_tokens)
    for traj in scoring_data.get_all_trajectories()
]

# Compute dynamics
scorer = Scorer(config)  # or pass config directly
result = compute_dynamics(trajectories, scorer, step=4)

# Save dynamics data as JSON
save_dynamics_json(result, Path("out/.../dynamics.json"))

# Generate visualization plots
saved_paths = plot_dynamics(result, Path("out/.../viz/dynamics/"))
```

## Metrics

At each token position k, we score the partial text and compute:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Pull** | x(k) | L2 norm of scores at position k (normative strength) |
| **Drift** | y(k) | Deviance from initial scores (how far from start) |
| **Potential** | z(k) | Deviance from final scores (how far to end state) |

## Algorithm

1. For each trajectory, identify measurement positions: `step, 2*step, ..., n_tokens`
2. At each position k:
   - Extract partial text (proportional to k/n_tokens)
   - Score it using the scoring config to get structure scores
   - Compute pull = `||scores||` (L2 norm)
   - Compute drift = `||scores - initial_scores||` (deviance from position 0)
   - Compute potential = `||scores - final_scores||` (deviance from end)

## Key Files

| File | Purpose |
|------|---------|
| `dynamics_computation.py` | `compute_dynamics()` - main computation |
| `dynamics_types.py` | `PositionScores`, `TrajectoryDynamics`, `DynamicsResult` |
| `dynamics_visualization.py` | `plot_dynamics()` - per-trajectory plots |
| `dynamics_serialization.py` | `save_dynamics_json()` - save to JSON |

## Output Files

```
out/<method>/<gen_name>/<scoring_name>/
    dynamics.json                    # JSON with (k, value) pairs
    viz/dynamics/
        all/                         # Individual trajectory plots
            traj_0_trunk.png
            traj_1_branch_1.png
            ...
        dynamics_trunk.png           # Aggregate: all trunk trajectories overlaid
        dynamics_branch_1.png        # Aggregate: all branch_1 trajectories overlaid
        dynamics_branch_2.png        # Aggregate: all branch_2 trajectories overlaid
```

### dynamics.json Format

```json
{
  "n_structures": 4,
  "step": 4,
  "trajectories": [
    {
      "traj_idx": 0,
      "arm_name": "trunk",
      "n_tokens": 64,
      "pull": [[4, 1.12], [8, 1.62], ...],
      "drift": [[4, 0.0], [8, 0.8], ...],
      "potential": [[4, 1.5], [8, 1.2], ...]
    }
  ]
}
```

## Trajectory Selection

Dynamics analyzes **extremal trajectories** - those with highest and lowest inverse perplexity per arm. The selection algorithm:

1. Filter to arm types in `DYNAMICS_ARMS` (e.g., only branches)
2. Group trajectories by arm name (e.g., `branch_1`, `branch_2`)
3. Sort each group by inverse perplexity
4. Pick `DYNAMICS_TRAJS_PER_ARM` most extremal, alternating from low and high ends

**String selection**: The same `STRING_SELECTION` from `default_config.py` is applied (e.g., `NonThinkingContinuation` strips `<think>...</think>` blocks).

## Common Pitfalls

1. **Re-scoring at each position** - partial text is re-scored at each k
2. **Step size matters** - smaller step = more measurements but slower
3. **Drift starts at 0** - by definition, drift from initial scores is 0 at first position
4. **Potential ends at 0** - by definition, potential to final scores is 0 at last position
5. **String selection applies** - thinking blocks are stripped before dynamics scoring (same as main scoring)
6. **Extremal selection** - not all trajectories are analyzed, only the most/least probable per arm

## See Also

- [../estimation/CLAUDE.md](../estimation/CLAUDE.md) - estimation module
- [Root CLAUDE.md](../../CLAUDE.md) - global project rules

## Workflow Orchestration

1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
