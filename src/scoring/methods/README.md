# Scoring Methods

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Implementation of trajectory scoring methods.

## Contents

- `categorical_method.py` - Binary yes/no judgments using LLM
- `graded_method.py` - Continuous 0-1 scale judgments using LLM
- `similarity_method.py` - Embedding-based cosine similarity
- `logging/` - Method-specific logging utilities

## Method Overview

### Categorical

Ask the judge model to answer 0 (no) or 1 (yes):

```
Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT: {trajectory_text}

QUESTION: Does this text mention a person?

Answer with just 0 or 1:
```

### Graded

Ask the judge model to score on a 0.0 to 1.0 scale:

```
Read the following text and answer the question with a score between 0.0 and 1.0.

TEXT: {trajectory_text}

QUESTION: How masculine is the protagonist?

Answer with just a number between 0.0 and 1.0:
```

### Similarity

Compute cosine similarity between trajectory embedding and reference word embeddings.
