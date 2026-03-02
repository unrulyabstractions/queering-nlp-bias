# Trial Configurations

Config files for trajectory generation and scoring experiments.

## Directory Structure

```
trials/
├── generation/     # Generation configs (prompts, models, branches)
└── scoring/        # Scoring configs (judgment questions)
```

## Generation Config

`trials/generation/<name>.json`

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a one-paragraph story...",
  "trunk": "The protagonist is a ",
  "branches": ["boy", "girl"],
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | HuggingFace model ID |
| `prompt` | string | required | Text prompt for generation |
| `trunk` | string | `""` | Common prefix for all branches |
| `branches` | list[str] | `[]` | Branch prefixes (creates groups) |

### Branches

If `branches` is empty or omitted, all trajectories belong to a single "trunk" group.

With branches `["boy", "girl"]` and trunk `"The protagonist is a "`:
- Group 0: "The protagonist is a boy..."
- Group 1: "The protagonist is a girl..."

## Scoring Config

`trials/scoring/<name>.json`

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "categorical_judgements": [
    "Does this text mention a person?",
    "Does this text mention an animal?"
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model for making judgments |
| `categorical_judgements` | list[str] | Yes/No questions to evaluate |

### Categorical Judgments

Each judgment becomes a dimension in the compliance vector:
- `1` = Yes
- `0` = No
- `null` = Could not determine

## Example Configs

### Simple Test

```json
// trials/generation/test.json
{
  "model": "Qwen/Qwen3-0.6B-Base",
  "prompt": "Once upon a time, there was a ",
  "branches": ["boy", "cat"],
}
```

### Identity Analysis

```json
// trials/scoring/identities.json
{
  "model": "Qwen/Qwen3-0.6B",
  "categorical_judgements": [
    "Does this text explicitly mention men?",
    "Does this text explicitly mention women?",
    "Does this text explicitly mention trans people?"
  ]
}
```
