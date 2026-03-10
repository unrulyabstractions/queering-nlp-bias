# Scoring Configs

Configuration files for trajectory scoring/judgment.

## Format

```json
{
  "model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    "Does this mention X?",
    ["Question A?", "Question B?"]
  ],
  "graded_judgements": [
    "How X is this? (0-1)"
  ],
  "similarity_scoring": [
    "reference_word",
    ["word1", "word2"]
  ]
}
```

## Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes* | - | Judge model (instruction-tuned) |
| `categorical_judgements` | No | `[]` | Binary yes/no questions |
| `graded_judgements` | No | `[]` | Scale questions (0.0-1.0) |
| `similarity_scoring` | No | `[]` | Embedding similarity references |
| `embedding_model` | No | `"all-MiniLM-L6-v2"` | Model for similarity |
| `string_selection` | No | `"WholeContinuation"` | What text to score |

*Required if using judgements.

## Grouping

Questions in arrays are grouped and averaged into a single structure:
```json
["Does this mention a person?", "Does this mention a human?"]
```

See `example.json` for a complete example.
