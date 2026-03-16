# Scoring Methods

Pluggable scoring methods for trajectories. All methods are automatically registered and discovered by the pipeline.

## Available Methods

### Categorical (`categorical_method.py`)
- **Type**: LLM-based, binary judgment
- **Output**: 0 or 1
- **Config key**: `categorical_judgements`
- **Label prefix**: `c` (c1, c2, c3, ...)
- **Requirements**: Model runner
- **Scoring**: Prompts the judge model with yes/no questions

### Graded (`graded_method.py`)
- **Type**: LLM-based, continuous scale
- **Output**: Float 0.0-1.0
- **Config key**: `graded_judgements`
- **Label prefix**: `g` (g1, g2, g3, ...)
- **Requirements**: Model runner
- **Scoring**: Prompts the judge model for nuanced 0.0-1.0 scale judgments

### Similarity (`similarity_method.py`)
- **Type**: Embedding-based, no LLM
- **Output**: Float 0.0-1.0 (cosine similarity)
- **Config key**: `similarity_scoring`
- **Label prefix**: `s` (s1, s2, s3, ...)
- **Requirements**: Embedding model
- **Scoring**: Computes cosine similarity between text embedding and reference embeddings

### Count Occurrences (`count_occurrences_method.py`)
- **Type**: Lexical, no LLM
- **Output**: Float (ratio)
- **Config key**: `count_occurrences`
- **Label prefix**: `o` (o1, o2, o3, ...)
- **Requirements**: None
- **Scoring**: Returns (# occurrences) / (# total words) for each target word/phrase

### Whistles (`whistles_method.py`)
- **Type**: LLM-based, glossary + judgment
- **Output**: Float 0.0-1.0 (aggregate probability of coded language)
- **Config key**: `whistles`
- **Label prefix**: `w` (w1)
- **Requirements**: Model runner, glossary JSON file
- **Scoring**: Finds glossary term matches, prompts LLM for P(coded|term,context), aggregates via noisy-OR or max
- **Reference**: Mendelsohn et al. (ACL 2023) - aclanthology.org/2023.acl-long.845

### Marked Personas (`marked_personas_method.py`)
- **Type**: Two-phase (LLM generation + lexicon scoring)
- **Output**: Float 0.0-1.0 (fraction of marked language signal)
- **Config key**: `marked_personas`
- **Label prefix**: `p` (p1)
- **Requirements**: Model runner (Phase 1 only), lexicon file (cached)
- **Scoring**: Phase 1 generates marked/unmarked personas to build z-score lexicon; Phase 2 scores text via dictionary lookup
- **Reference**: Cheng et al. (ACL 2023) - aclanthology.org/2023.acl-long.84

## Method Registry

All methods are auto-discovered. To add a new method:

1. Create a file named `my_method_name.py`
2. Define a params dataclass inheriting from `ScoringMethodParams`:
   ```python
   @dataclass
   class MyParams(ScoringMethodParams):
       name: ClassVar[str] = "my-method"
       config_key: ClassVar[str] = "my_method_config"
       label_prefix: ClassVar[str] = "m"
       requires_runner: ClassVar[bool] = False
       requires_embedder: ClassVar[bool] = False
   ```
3. Define a scoring function and decorate with `@register_method(MyParams)`:
   ```python
   @register_method(MyParams)
   def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
       # items = list[str | list[str]] from config[config_key]
       # Return: (scores, raw_responses) tuple
       ...
   ```
4. No other changes needed — the method is automatically discovered and integrated into the pipeline

**Shared utilities**: Use `score_with_bundling()` from `scoring_method_registry` to handle bundled items and logging consistently.
