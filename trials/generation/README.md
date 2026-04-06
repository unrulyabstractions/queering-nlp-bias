# Generation Configs

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Configuration files for trajectory generation experiments.

## Format

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "Write a story...",
  "trunk": "Once upon a time",
  "branches": [" boy", " girl"],
  "twig_variations": [" and his dog", " and her dog"],
  "temperature": 1.0,
  "max_new_tokens": 128
}
```

## Fields

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `model` | Yes | - | HuggingFace model ID or API provider |
| `prompt` | Yes\* | - | System/instruction prompt (\*not used in template mode) |
| `trunk` | No | `""` | Shared prefix for all trajectories |
| `branches` | No | `[]` | Branch-specific prefixes (appended after trunk) |
| `twig_variations` | No | `[]` | Per-branch suffixes; each branch × twig becomes one arm |
| `temperature` | No | `1.0` | Sampling temperature |
| `max_new_tokens` | No | `128` | Maximum tokens to generate |
| `seed` | No | `null` | Random seed for reproducibility |
| `prompt_template` | No\*\* | - | Prompt with `{word}` placeholder (template mode) |
| `template_words` | No\*\* | - | Words substituted into `prompt_template`, one arm each |

\*\* `prompt_template` and `template_words` must be used together and are mutually exclusive with `prompt`, `trunk`, `branches`, and `twig_variations`.
| `method_params` | No | `{}` | Method-specific parameter overrides |

## Method Parameters

Override method-specific settings via `method_params`:

```json
{
  "model": "Qwen/Qwen3-0.6B",
  "prompt": "...",
  "method_params": {
    "simple-sampling": {
      "overrides": {"samples_per_arm": 20}
    },
    "forking-paths": {
      "overrides": {
        "max_alternates": 5,
        "min_prob": 0.15,
        "samples_per_fork": 3
      }
    }
  }
}
```

| Method | Available Overrides |
|--------|---------------------|
| `simple-sampling` | `samples_per_arm` |
| `forking-paths` | `max_alternates`, `min_prob`, `min_entropy`, `samples_per_fork` |
| `seeking-entropy` | `samples_per_expansion`, `num_expansion_rounds` |

See `example.json` for a complete example.
