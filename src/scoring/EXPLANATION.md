# Scoring Package Specification

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


This document provides detailed specification for the trajectory scoring system.

## Overview

The scoring package evaluates generated text trajectories against configurable criteria. It supports multiple scoring methods through a registry pattern that allows adding new methods with minimal code changes.

## Architecture

```
ScoringConfig (configuration)
       |
       v
run_scoring_pipeline() ---------> ScoringPipelineResult
       |                                  |
       v                                  v
score_trajectory()              ScoringOutput (serializable)
       |                                  |
       v                                  v
get_method() --> score_fn()     ScoringResult (per-trajectory)
```

## Scoring Methods

### Categorical Method

**Purpose**: Binary yes/no judgments using an LLM as judge.

**Config Key**: `categorical_judgements`
**Label Prefix**: `c` (produces c1, c2, ...)
**Requires**: LLM runner

**How it works**:
1. For each question, builds a prompt asking the LLM to judge with 0 (no) or 1 (yes)
2. Sends prompt to the judge model with temperature=0
3. Parses response to extract 0 or 1

**Prompt format**:
```
Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT:
{trajectory_text}

QUESTION: {question}

Answer with just 0 or 1:
```

**Response parsing** (in order of precedence):
1. Strips `</think>` blocks if present
2. Exact match: "0" or "1"
3. Pattern: `answer: 0` or `result: 1`
4. Word match: "yes" -> 1, "no" -> 0
5. Trailing digit: `... 1`
6. Any digit: first `0` or `1` found

**Output**: `int | None` (None if parsing fails)

**Parameters**:
- `max_tokens`: Maximum tokens for LLM response (default: 10)

---

### Graded Method

**Purpose**: Continuous 0.0-1.0 scale judgments using an LLM as judge.

**Config Key**: `graded_judgements`
**Label Prefix**: `g` (produces g1, g2, ...)
**Requires**: LLM runner

**How it works**:
1. For each question, builds a prompt asking for a 0.0-1.0 score
2. Sends prompt to the judge model with temperature=0
3. Parses response to extract a float

**Prompt format**:
```
Read the following text and answer the question with a score between 0.0 and 1.0.
0.0 means completely no/false, 1.0 means completely yes/true, values in between indicate partial agreement.

TEXT:
{trajectory_text}

QUESTION: {question}

Answer with just a number between 0.0 and 1.0:
```

**Response parsing**:
1. Strips `</think>` blocks if present
2. Searches for decimal pattern: `0.5`, `1.0`, `.75`, etc.
3. Validates the value is in [0.0, 1.0]

**Output**: `float | None` (None if parsing fails)

**Parameters**:
- `max_tokens`: Maximum tokens for LLM response (default: 10)

---

### Similarity Method

**Purpose**: Embedding-based semantic similarity scoring.

**Config Key**: `similarity_scoring`
**Label Prefix**: `s` (produces s1, s2, ...)
**Requires**: Embedding runner

**How it works**:
1. Embeds the trajectory text using the configured embedding model
2. Embeds each reference string
3. Computes cosine similarity between trajectory and each reference

**Output**: `float` (always produces a value, typically in range [-1, 1] but often [0, 1])

**Parameters**:
- `embedding_model`: Model name for sentence embeddings (default: "all-MiniLM-L6-v2")

**Note**: All references are embedded in a single batch for efficiency.

---

### Count Occurrences Method

**Purpose**: Simple word/phrase frequency counting.

**Config Key**: `count_occurrences`
**Label Prefix**: `o` (produces o1, o2, ...)
**Requires**: Nothing (pure text processing)

**How it works**:
1. Counts total words in the trajectory text
2. For each target word/phrase, counts occurrences
3. Returns ratio: `occurrences / total_words`

**Matching behavior**:
- Single words: Uses word boundaries (`\bword\b`)
- Phrases with spaces: Exact substring match
- Case sensitivity: Configurable (default: case-insensitive)

**Output**: `float` (ratio, typically very small values like 0.001)

**Parameters**:
- `case_sensitive`: Whether matching is case-sensitive (default: false)

---

## Data Flow

### Input Processing

1. **Load Generation Output**:
   ```python
   gen_data = GenerationOutputData.load("out/gen_example.json")
   ```
   This extracts `TrajectoryData` objects containing:
   - `trajectory_idx`: Index in the generation batch
   - `branch`: Which branch the trajectory came from (e.g., "trunk", "branch_1")
   - `prompt`: The input text
   - `response`: The generated continuation
   - `response_after_branch`: Continuation with branch token stripped
   - `conditional_logprobs`: Log probabilities conditioned on each arm

2. **Text Selection**:
   The `string_selection` config option determines which text portion to score:
   - `WholeTrajectory`: prompt + response
   - `WholeContinuation`: just the response (default)
   - `AfterTrunk`: response minus trunk
   - `AfterBranch`: response minus branch token

3. **EOS Token Stripping**:
   The pipeline strips EOS tokens from text before scoring to avoid artifacts.

### Scoring Execution

For each trajectory:
1. Extract text based on `string_selection`
2. For each active method (methods with configured items):
   - Get method function from registry
   - Get method parameters (defaults + overrides)
   - Call method with text and items
   - Store (scores, raw_responses)

### Output Structure

**Per-trajectory** (`ScoringResult`):
```python
{
    "trajectory_idx": 0,
    "branch": "trunk",
    "branch_idx": 0,
    "text": "...",
    "conditional_logprobs": {"trunk": -5.2, "branch_1": -8.1},
    "n_continuation_tokens": 45,
    "method_scores": {
        "categorical": [1, 0, 1],
        "graded": [0.75],
        "similarity": [0.82, 0.45]
    },
    "method_raw": {
        "categorical": ["1", "0", "1"],
        "graded": ["0.75"],
        "similarity": ["", ""]
    }
}
```

**Full output** (`ScoringOutput`):
```python
{
    "generation_file": "out/gen_example.json",
    "scoring_file": "trials/scoring/example.json",
    "judge_model": "gpt-4o-mini",
    "embedding_model": "all-MiniLM-L6-v2",
    "scoring_data": {
        "categorical_judgements": ["Q1?", "Q2?"],
        "similarity_scoring": ["reference text"]
    },
    "branches": ["trunk", "branch_1"],
    "scored_at": "2024-01-15T10:30:00",
    "num_results": 100,
    "results": [...]
}
```

---

## Registry Pattern

The registry allows adding new scoring methods with a single file. No changes to core pipeline code are needed.

### How Registration Works

1. Define a params class inheriting from `ScoringMethodParams`:
   ```python
   @dataclass
   class MyParams(ScoringMethodParams):
       name: ClassVar[str] = "my-method"          # Registry lookup key
       config_key: ClassVar[str] = "my_method_data"  # JSON config field
       label_prefix: ClassVar[str] = "m"          # Structure labels (m1, m2...)
       requires_runner: ClassVar[bool] = False    # Needs LLM?
       requires_embedder: ClassVar[bool] = False  # Needs embeddings?

       # Method-specific parameters
       my_param: int = 10
   ```

2. Decorate the scoring function with `@register_method`:
   ```python
   @register_method(MyParams)
   def score_my_method(
       text: str,
       items: list[str | list[str]],
       params: MyParams,
       runner: ModelRunner | None = None,
       embedder: EmbeddingRunner | None = None,
       log_fn: LogFn | None = None,
   ) -> tuple[list[Any], list[str]]:
       # Return (scores, raw_responses)
       ...
   ```

3. The method is automatically discovered and available in configs.

### Registry API

```python
from src.scoring.scoring_method_registry import (
    list_methods,      # Returns: ["categorical", "graded", "similarity", "count-occurrences"]
    get_method,        # Returns the scoring function
    get_params_class,  # Returns the params class
    get_default_params, # Returns params instance with defaults
    iter_methods,      # Returns (name, params_class, score_fn) tuples
)
```

---

## Bundled vs Single Items

Items in the config can be either single strings or lists of strings (bundles).

### Single Items

Each string becomes one "structure" with one score:

```json
{
  "categorical_judgements": [
    "Does this mention a person?",
    "Does this describe an action?"
  ]
}
```
Produces: c1 (one score), c2 (one score)

### Bundled Items

A list of strings becomes one "structure" with multiple scores that are averaged:

```json
{
  "categorical_judgements": [
    ["Does this mention a person?", "Does this mention a human?"],
    "Does this describe an action?"
  ]
}
```
Produces: c1 (average of two scores), c2 (one score)

**Why use bundles?**
- Combine semantically related questions
- Reduce noise from individual judgment variance
- Create composite metrics (e.g., "mentions any animal" = avg of "cat?", "dog?", "bird?")

### How Scores Are Computed

For single items:
```
score = value from scoring function
```

For bundled items:
```
score = mean(value1, value2, ...)
```

The flat list of individual scores is stored in `method_scores`, while the structure-level aggregation happens at summary/display time.

---

## Configuration Options

### ScoringConfig Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | "" | LLM for judge-based methods |
| `embedding_model` | str | "all-MiniLM-L6-v2" | Model for similarity scoring |
| `string_selection` | StringSelection | WholeContinuation | Which text portion to score |
| `max_tokens` | int | 10 | Max tokens for LLM responses |
| `method_params` | dict | {} | Per-method parameter overrides |
| `scoring_data` | dict | {} | Method items (auto-populated from flat config keys) |

### Example Config

```json
{
  "model": "gpt-4o-mini",
  "embedding_model": "all-MiniLM-L6-v2",
  "string_selection": "WholeContinuation",
  "max_tokens": 15,

  "categorical_judgements": [
    "Does this text mention a specific person by name?",
    ["Does the text include dialogue?", "Does someone speak in quotes?"]
  ],

  "graded_judgements": [
    "How formal is the writing style?"
  ],

  "similarity_scoring": [
    "scientific research paper",
    "casual conversation"
  ],

  "count_occurrences": [
    "the",
    ["he", "she", "they"]
  ],

  "method_params": {
    "categorical": {
      "overrides": {
        "max_tokens": 20
      }
    },
    "count-occurrences": {
      "overrides": {
        "case_sensitive": true
      }
    }
  }
}
```

### String Selection Options

| Value | Description |
|-------|-------------|
| `WholeTrajectory` | Full text including prompt and response |
| `WholeContinuation` | Just the generated response (default) |
| `AfterTrunk` | Response minus trunk tokens |
| `AfterBranch` | Response minus branch token |

---

## Helper Utilities

### score_with_bundling

A shared utility for methods that process items one at a time:

```python
from src.scoring.scoring_method_registry import score_with_bundling

def my_score_fn(text, items, params, ...):
    def score_single(item: str) -> tuple[Any, str]:
        # Process one item
        return (score, raw_response)

    return score_with_bundling(items, score_single, params.label_prefix, log_fn)
```

This handles:
- Iterating over items (single or bundled)
- Logging with proper prefixes
- Building the flat output lists

### Logging

Methods receive a `log_fn` callback for progress reporting:

```python
log_fn("[c1] Does this mention...? -> 1")
log_fn("[c2] Bundled (3 items)")
log_fn("     * Question 1? -> 0")
log_fn("     * Question 2? -> 1")
```

---

## Adding a New Method

1. Create `src/scoring/methods/my_method.py`:

```python
from dataclasses import dataclass
from typing import ClassVar
from ..scoring_method_registry import ScoringMethodParams, register_method

@dataclass
class MyParams(ScoringMethodParams):
    name: ClassVar[str] = "my-method"
    config_key: ClassVar[str] = "my_method_items"
    label_prefix: ClassVar[str] = "m"
    requires_runner: ClassVar[bool] = False
    requires_embedder: ClassVar[bool] = False

    threshold: float = 0.5  # Method-specific param

@register_method(MyParams)
def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
    scores = []
    for item in items:
        if isinstance(item, list):
            for sub in item:
                scores.append(compute_score(text, sub, params.threshold))
        else:
            scores.append(compute_score(text, item, params.threshold))
    return scores, [""] * len(scores)
```

2. The method is automatically available in configs:

```json
{
  "my_method_items": ["item1", "item2"],
  "method_params": {
    "my-method": {
      "overrides": {"threshold": 0.7}
    }
  }
}
```

No other files need modification.
