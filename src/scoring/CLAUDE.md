# CLAUDE.md - src/scoring/

This module scores generated trajectories using configurable methods (LLM judgments, similarity, word counts).

## Architecture

```python
# Simple: one object, simple methods
scorer = Scorer(config)        # or Scorer.load("config.json")
scores = scorer.score(text)    # -> list[float] (structure scores)

# Pipeline for many trajectories
result = run_scoring_pipeline(config, trajectories, arm_names, arm_texts)
```

`Scorer` mirrors `ModelRunner`: load once, call many times.

## Registry Pattern

**Adding a new scoring method requires ONE FILE in `methods/`.**

```python
# methods/my_scoring_method.py
from dataclasses import dataclass
from typing import ClassVar
from ..scoring_method_registry import ScoringMethodParams, register_method

@dataclass
class MyParams(ScoringMethodParams):
    name: ClassVar[str] = "my-method"           # Registry key
    config_key: ClassVar[str] = "my_method_items"  # JSON field name
    label_prefix: ClassVar[str] = "m"           # Labels: m1, m2...
    requires_runner: ClassVar[bool] = False     # Needs LLM?
    requires_embedder: ClassVar[bool] = False   # Needs embeddings?

    threshold: float = 0.5  # Method-specific param

@register_method(MyParams)
def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
    scores = []
    raw_responses = []
    for item in items:
        # Score each item
        scores.append(compute_score(text, item, params.threshold))
        raw_responses.append("")
    return scores, raw_responses
```

The method is automatically discovered - no other changes needed.

## Key Files

| File | Purpose |
|------|---------|
| `scoring_pipeline.py` | `run_scoring_pipeline()` entry point |
| `scoring_method_registry.py` | `@register_method`, `get_method()`, `score_with_bundling()` |
| `scoring_config.py` | Config loading with method params |
| `scoring_data.py` | `TrajectoryData`, `GenerationOutputData` input types |
| `scoring_output.py` | `ScoringOutput`, `ScoringResult` output types |

## Available Methods

| Method | Config Key | Label | Requires |
|--------|------------|-------|----------|
| `categorical` | `categorical_judgements` | c1, c2... | LLM |
| `graded` | `graded_judgements` | g1, g2... | LLM |
| `similarity` | `similarity_scoring` | s1, s2... | Embedder |
| `count-occurrences` | `count_occurrences` | o1, o2... | None |
| `whistles` | `whistles` | w1 | LLM |
| `marked_personas` | `marked_personas` | p1 | LLM (Phase 1 only) |

### Whistles (Dog Whistle Detection)

Based on Mendelsohn et al. (ACL 2023). Detects coded language using a glossary lookup + LLM judgment.

```json
{
  "model": "anthropic/claude-sonnet",
  "whistles": ["_"],
  "method_params": {
    "whistles": {
      "glossary_path": "path/to/glossary.json",
      "aggregation": "noisy_or"
    }
  }
}
```

Glossary format:
```json
[
  {
    "surface_form": "confirmed bachelor",
    "covert_meaning": "gay man",
    "type": "II",
    "example_coded": "He was a confirmed bachelor.",
    "example_literal": "He was a bachelor looking for a wife."
  }
]
```

Process: Find matches → LLM estimates P(coded|term,context) → Aggregate via noisy-OR or max.

### Marked Personas

Based on Cheng et al. (ACL 2023). Measures how much text contains language LLMs associate with a marked group.

```json
{
  "model": "anthropic/claude-sonnet",
  "marked_personas": ["_"],
  "method_params": {
    "marked_personas": {
      "marked_label": "queer",
      "domain": "software engineer",
      "lexicon_path": "out/lexicons/queer_swe.json",
      "n_samples": 100
    }
  }
}
```

Two phases:
- **Phase 1** (expensive, done once): Generate N marked + N unmarked personas, compute Fightin' Words z-scores
- **Phase 2** (instant): Score text using precomputed lexicon

If `lexicon_path` exists, Phase 1 is skipped. First run takes time, subsequent runs are fast.

## Bundled Items

Config items can be single strings or lists (bundles):

```json
{
  "categorical_judgements": [
    "Simple question?",
    ["Bundled Q1?", "Bundled Q2?"]  // Averaged into one structure score
  ]
}
```

Use `score_with_bundling()` from registry to handle this automatically.

## String Selection

The `string_selection` config determines which text portion to score:
- `WholeContinuation` - all arm prefills + generated text (`<think>` blocks kept)
- `NonThinkingContinuation` - all arm prefills + generated text, `<think>` blocks removed **(default)**
- `AfterTrunk` - text after trunk arm prefix (`<think>` stripped)
- `AfterBranch` - text after branch arm prefix (`<think>` stripped)
- `AfterTwig` - raw generated text only, no arm prefills (`<think>` blocks kept)
- `WholeTrajectory` - not dispatched; falls through to `NonThinkingContinuation`

## Common Pitfalls

1. **Methods must return `(scores, raw_responses)` tuple** - both lists
2. **Bundle support is optional** - use `score_with_bundling()` helper
3. **Config key must match JSON field name** - e.g., `my_method_items` not `myMethodItems`
4. **All params need defaults** - for `get_default_params()` to work

## Output Path Convention

```
out/<gen-method>/<gen-name>/<scoring-name>/scoring.json
out/<gen-method>/<gen-name>/<scoring-name>/summary_scoring.txt
```

## See Also

- [EXPLANATION.md](./EXPLANATION.md) - detailed algorithm specification
- [ADDING_METHOD.md](./ADDING_METHOD.md) - step-by-step guide
- [methods/README.md](./methods/README.md) - existing method implementations
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
