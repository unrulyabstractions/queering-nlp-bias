# Trial Configurations

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Config files for trajectory generation and scoring experiments.

## Directory Structure

```
trials/
├── generation/     # Generation configs (prompts, models, branches)
│   ├── example.json
│   ├── boxer.json
│   ├── decision.json
│   └── ...
└── scoring/        # Scoring configs (judgment questions)
    ├── example.json
    ├── identities.json
    └── ...
```

## Generation Config Format

`trials/generation/<name>.json`

### Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | - | HuggingFace model ID or API provider |
| `prompt` | string | yes\* | - | System/user prompt for generation (\*not used in template mode) |
| `trunk` | string | no | `""` | Shared prefix for all branches |
| `branches` | list[str] | no | `[]` | Branch-specific prefixes (appended after trunk) |
| `twig_variations` | list[str] | no | `[]` | Per-branch suffixes; each branch × twig becomes one arm |
| `temperature` | float | no | `1.0` | Sampling temperature |
| `max_new_tokens` | int | no | `128` | Maximum tokens to generate |
| `seed` | int | no | `null` | Random seed for reproducibility |
| `method_params` | object | no | `{}` | Method-specific parameter overrides |
| `prompt_template` | string | no\*\* | - | Prompt with `{word}` placeholder (template mode) |
| `template_words` | list[str] | no\*\* | - | Words substituted into `prompt_template`, one arm each |

\*\* `prompt_template` and `template_words` must be used together and are mutually exclusive with `prompt`, `trunk`, `branches`, and `twig_variations`.

### Method Parameter Overrides

Override generation method settings via `method_params`:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "...",
  "method_params": {
    "simple-sampling": {
      "overrides": {"samples_per_arm": 20}
    }
  }
}
```

| Method | Available Overrides |
|--------|---------------------|
| `simple-sampling` | `samples_per_arm` |
| `forking-paths` | `max_alternates`, `min_prob`, `min_entropy`, `samples_per_fork` |
| `seeking-entropy` | `samples_per_expansion`, `num_expansion_rounds` |

### Branching Logic

Arms are built in a hierarchy: **root → trunk → branches → twigs**.

- If `branches` is empty: only the root (and trunk if set) arms are generated.
- If `branches` has entries: each branch becomes one arm (prefill = trunk + branch text).
- If `twig_variations` is set: each branch spawns one arm per twig (prefill = trunk + branch + twig), creating `len(branches) × len(twig_variations)` extra arms.

**Example (branches only)** — `trunk: "The protagonist is a"`, `branches: [" boy", " girl"]`:
- arm `root`: trunk prefix only
- arm `branch_1`: "The protagonist is a boy"
- arm `branch_2`: "The protagonist is a girl"

**Example (branches + twigs)** — `trunk: "At the altar, our protagonist, who was a"`, `branches: [" mechanic,", " nurse,"]`, `twig_variations: [" and his partner", " and her partner"]`:
- arm `branch_1`: "…a mechanic,"
- arm `twig_b1_1`: "…a mechanic, and his partner"
- arm `twig_b1_2`: "…a mechanic, and her partner"
- arm `branch_2`: "…a nurse,"
- arm `twig_b2_1`: "…a nurse, and his partner"
- arm `twig_b2_2`: "…a nurse, and her partner"

### Template Mode

An alternative to trunk/branches for simple substitution experiments. Set `prompt_template` (a string containing `{word}`) and `template_words` (a list of substitution values). Each word produces one arm with its own filled prompt and no prefill text.

Template mode is **mutually exclusive** with `prompt`, `trunk`, `branches`, and `twig_variations`.

```json
{
  "model": "openai/gpt-4o-mini",
  "prompt_template": "Write a one-sentence bio for a {word}.",
  "template_words": ["nurse", "engineer", "teacher"]
}
```

### Examples

**Minimal**:
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Once upon a time, there was a "
}
```

**With Branches**:
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a story in LESS than 12 words about EITHER a boy OR a cat.",
  "trunk": "Once upon a time, there was a",
  "branches": [" boy", " cat"]
}
```

**With Branches + Twigs**:
```json
{
  "model": "Qwen/Qwen3.5-2B",
  "prompt": "In less than three sentences, write an extremely short love story.",
  "trunk": "At the altar, our protagonist, who was a ",
  "branches": ["mechanic,", "nurse,"],
  "twig_variations": [" and his partner", " and her partner"]
}
```

**Template Mode**:
```json
{
  "model": "openai/gpt-4o-mini",
  "prompt_template": "Generate a persona of a {word}.",
  "template_words": ["man", "woman", "non-binary person"]
}
```

**Narrative Experiment** (`boxer.json`):
```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a one-paragraph story in which two people meet, fall in love, and live happily ever after.",
  "trunk": "The tough boxer is also a biker who loves ",
  "branches": ["drag queens", "drag racing"]
}
```

---

## Scoring Config Format

`trials/scoring/<name>.json`

### Schema

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes* | - | Judge model (required if using judgements) |
| `categorical_judgements` | list | no | `[]` | Binary yes/no questions |
| `graded_judgements` | list | no | `[]` | Scale questions (0.0 to 1.0) |
| `similarity_scoring` | list | no | `[]` | Words/phrases for embedding similarity |
| `string_selection` | enum | no | `"WholeContinuation"` | Which text portion to score |
| `max_tokens` | int | no | `64` | Judge response token limit |
| `embedding_model` | string | no | `"all-MiniLM-L6-v2"` | Model for similarity scoring |

*Required if `categorical_judgements` or `graded_judgements` is non-empty.

### Structure Types

#### 1. Categorical Judgements

Binary yes/no questions. Each question becomes a structure with values 0 or 1.

```json
{
  "categorical_judgements": [
    "Does this text mention a woman?",
    "Does this text mention a man?"
  ]
}
```

**Grouped Questions**: Questions in a sub-list are averaged into a single structure value:

```json
{
  "categorical_judgements": [
    ["Does this mention a person?", "Does this mention a boy?"],
    "Does this mention happiness?"
  ]
}
```
- `c1`: Average of the two "person/boy" questions
- `c2`: Single "happiness" question

#### 2. Graded Judgements

Questions scored on a continuous 0.0 to 1.0 scale:

```json
{
  "graded_judgements": [
    "How masculine is the protagonist?",
    ["How happy is the ending?", "How satisfying is the resolution?"]
  ]
}
```

#### 3. Similarity Scoring

Embedding similarity to reference words/phrases:

```json
{
  "similarity_scoring": [
    "feminine",
    ["gay", "queer", "LGBTQ+"]
  ]
}
```
- `s1`: Cosine similarity to "feminine"
- `s2`: Average similarity to "gay", "queer", "LGBTQ+"

### String Selection Options

| Value | Description |
|-------|-------------|
| `WholeTrajectory` | Full text including prompt and response |
| `WholeContinuation` | Just the generated response (default) |
| `AfterTrunk` | Continuation minus trunk text |
| `AfterBranch` | Continuation minus trunk and branch |

### Examples

**Simple Binary**:
```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this text mention a person?",
    "Does this text mention an animal?"
  ]
}
```

**Full Suite** (`example.json`):
```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    ["Does this story talk about a person?", "Does this story talk about a boy?"],
    ["Does this story talk about an animal?", "Does this story talk about a cat?"],
    "Does this story talk about someone happy?"
  ],
  "graded_judgements": [
    ["How masculine is the protagonist?", "How non-feminine is the protagonist?"],
    "How sad is the protagonist?"
  ],
  "similarity_scoring": [
    ["gay", "queer"],
    "feline"
  ]
}
```

**Identity Analysis** (`identities.json`):
```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this text explicitly mention men?",
    "Does this text explicitly mention women?",
    "Does this text explicitly mention trans people?",
    "Does this text explicitly mention non-binary people?"
  ]
}
```

---

## Output Mapping

Given configs:
- Generation: `trials/generation/example.json`
- Scoring: `trials/scoring/example.json`

Outputs (with default simple-sampling method):
```
out/simple-sampling/gen_example.json           # Generated trajectories
out/simple-sampling/score_example_example.json # Judgment results
out/simple-sampling/est_example_example.json   # Estimation results
out/simple-sampling/summary_est_example_example.txt # Human-readable summary
```

With `--method forking-paths`:
```
out/forking-paths/gen_example.json
out/forking-paths/score_example_example.json
...
```
