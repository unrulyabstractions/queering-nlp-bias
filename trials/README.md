# Trial Configurations

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
| `prompt` | string | yes | - | System/user prompt for generation |
| `trunk` | string | no | `""` | Shared prefix for all branches |
| `branches` | list[str] | no | `[]` | Branch-specific prefixes |
| `temperature` | float | no | `1.0` | Sampling temperature |
| `max_new_tokens` | int | no | `128` | Maximum tokens to generate |
| `seed` | int | no | `null` | Random seed for reproducibility |

### Branching Logic

- If `branches` is empty: all trajectories share the trunk (single group)
- If `branches` has entries: creates one group per branch

**Example**: With `trunk: "The protagonist is a "` and `branches: [" boy", " girl"]`:
- Group 0 (trunk): "The protagonist is a " + continuation
- Group 1: "The protagonist is a boy" + continuation
- Group 2: "The protagonist is a girl" + continuation

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
out/gen_sampling_example.json           # Generated trajectories
out/score_sampling_example_example.json # Judgment results
out/est_sampling_example_example.json   # Estimation results
out/summary_est_sampling_example_example.json # Human-readable summary
```

With `--forking-paths`:
```
out/gen_forking_example.json
out/score_forking_example_example.json
...
```
