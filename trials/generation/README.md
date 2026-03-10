# Generation Configs

Configuration files for trajectory generation experiments.

## Format

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a story...",
  "trunk": "Once upon a time",
  "branches": [" boy", " girl"],
  "temperature": 1.0,
  "max_new_tokens": 128
}
```

## Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | - | HuggingFace model ID |
| `prompt` | Yes | - | System/instruction prompt |
| `trunk` | No | `""` | Shared prefix for all trajectories |
| `branches` | No | `[]` | Branch-specific prefixes |
| `temperature` | No | `1.0` | Sampling temperature |
| `max_new_tokens` | No | `128` | Maximum tokens to generate |
| `seed` | No | `null` | Random seed for reproducibility |

See `example.json` for a complete example.
