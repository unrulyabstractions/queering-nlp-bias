# CLAUDE.md - src/dynamics/

This module tracks how a trajectory evolves token by token and computes the paper's
**deviance-based** pull/drift/potential. At each measured prefix it tracks two distinct
quantities: the realized **system attunement** Λ_n(x_p) (score of the prefix) and the
**system default** ⟨Λ_n⟩(x_p) (the barycenter, estimated by **sampling continuations**
from the model and averaging their attunements — paper Eq. 7).

## Configuration

All dynamics parameters are defined in `src/common/default_config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DYNAMICS_STEP` | 8 | Measure the system default every N tokens (each measure samples → small step is expensive) |
| `DYNAMICS_SAMPLES_PER_POSITION` | 8 | Continuations sampled per prefix to estimate ⟨Λ_n⟩(x_p) |
| `DYNAMICS_CONTINUATION_MAX_TOKENS` | 128 | Max tokens per sampled continuation |
| `DYNAMICS_TRAJS_PER_ARM` | 2 | Number of most extremal trajectories to analyze per arm |
| `DYNAMICS_ARMS` | all four | Arm types to analyze: `"root"`, `"trunk"`, `"branch"`, `"twig"` |

## Main Entry Points

```python
from src.dynamics import (
    DynamicsConfig, DynamicsTrajectory,
    compute_dynamics, plot_dynamics, save_dynamics_json,
)
from src.inference import ModelRunner
from src.scoring import Scorer

scorer = Scorer.load(scoring_config_path)   # judge model → system attunement
runner = ModelRunner(gen_model_name)        # generation model → sampled continuations

# Each trajectory carries the prompt + arm prefill needed to sample continuations.
trajectories = [
    DynamicsTrajectory(
        traj_idx=t.traj_idx, arm_name=t.arm, prompt=prompt, prefill=arm_prefill,
        text=continuation, n_tokens=t.n_generated_tokens,
    )
    for t in selected_trajectories
]

config = DynamicsConfig(step=8, samples_per_position=8, continuation_max_tokens=128, temperature=1.0)
result = compute_dynamics(trajectories, scorer, runner, config)

save_dynamics_json(result, Path("out/.../dynamics.json"))
plot_dynamics(result, Path("out/.../viz/dynamics/"))
```

## Metrics

Every metric is an **orientation/deviance** (paper Eqs. 8-9): an *attunement* of a string
minus a *system default* of a reference prefix — never attunement−attunement or
default−default. All are dimension-normalized (`||·||_2 / sqrt(dim)`):

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Pull** | x(k) | `||⟨Λ_n⟩(x_p)||`, magnitude of the **system default** (normative strength) |
| **Drift** | y(k) | `||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||`, current **attunement** vs the **initial default** |
| **Potential** | z(k) | `||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||`, **final attunement** vs the **current default** |

## Algorithm

1. For each trajectory, identify measurement positions: `step, 2*step, ..., n_tokens`
2. At each position k:
   - Extract the prefix text (proportional to k/n_tokens)
   - Score it → the realized **system attunement** `Λ_n(x_p)`
   - **Sample `samples_per_position` continuations** from the prefix, score each, average
     → the **system default** `⟨Λ_n⟩(x_p)` (paper Eq. 7)
3. With per-position attunements and defaults, compute:
   - pull = `||⟨Λ_n⟩(x_p)||`
   - drift = `||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||` (uses the **initial** default ⟨Λ_n⟩(x_0))
   - potential = `||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||` (uses the **final** attunement Λ_n(x_final))

## Key Files

| File | Purpose |
|------|---------|
| `dynamics_computation.py` | `compute_dynamics()` - main computation/orchestration |
| `dynamics_sampling.py` | `estimate_system_default()` - sample continuations → barycenter ⟨Λ_n⟩ |
| `dynamics_metrics.py` | `pull()`, `drift()`, `potential()`, `normalized_norm()` |
| `dynamics_types.py` | `DynamicsTrajectory`, `DynamicsConfig`, `PositionScores`, `TrajectoryDynamics`, `DynamicsResult` |
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
  "step": 8,
  "trajectories": [
    {
      "traj_idx": 0,
      "arm_name": "trunk",
      "n_tokens": 64,
      "pull": [[8, 0.42], [16, 0.53], ...],
      "drift": [[8, 0.03], [16, 0.11], ...],
      "potential": [[8, 0.39], [16, 0.25], ...]
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

1. **Sampling cost dominates** - each position samples `samples_per_position` continuations
   from the model: cost ≈ `positions × samples_per_position` generations per trajectory.
   Raise `DYNAMICS_STEP` / lower `DYNAMICS_SAMPLES_PER_POSITION` to trade resolution for speed.
2. **Two models loaded at once** - the judge model (scorer) AND the generation model (runner)
   are resident together during dynamics; watch GPU memory.
3. **Drift does NOT start at 0** - drift = `||Λ_n(x_p) - ⟨Λ_n⟩(x_0)||`; at the first position
   this is the attunement's deviance from the default, which is generally nonzero.
4. **Potential does NOT end at 0** - potential = `||Λ_n(x_final) - ⟨Λ_n⟩(x_p)||`; at the last
   position the final attunement still deviates from the final prefix's default in general.
5. **Attunement ≠ default** - never conflate Λ_n(x_p) (realized prefix score) with ⟨Λ_n⟩(x_p)
   (sampled barycenter); they are tracked separately per position.
6. **temperature=1.0 for a valid barycenter** - the uniform mean over samples estimates
   E[Λ_n] only when sampling from the true distribution (temperature 1.0).
7. **String selection applies** - thinking blocks are stripped from both the analyzed text
   and each sampled continuation (same as main scoring).
8. **Extremal selection** - not all trajectories are analyzed, only the most/least probable per arm.

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
