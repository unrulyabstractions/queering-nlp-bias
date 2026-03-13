# Scoring Methodology

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

How we evaluate trajectories against structures.

## Conceptual Framework

A **structure** is a specification of organization among tokens - a question like "Does this text mention women?" We use **structure compliance** to score how well a string aligns with a structure:

```
α_i : Str → [0, 1]
```

Where `α_i(x) = 1` means full compliance, `α_i(x) = 0` means no compliance.

Multiple structures form a **system**. The **system compliance** is a vector:

```
Λ_n(x) = (α_1(x), ..., α_n(x))
```

## Scoring Types

### 1. Categorical Judgements

Binary yes/no questions scored as 0 or 1.

```json
{
  "categorical_judgements": [
    "Does this text mention a woman?",
    "Does this text mention violence?"
  ]
}
```

**How it works**:
1. Build a prompt asking the judge model to answer 0 or 1
2. Parse the response to extract the judgment
3. Handle edge cases (thinking tags, verbose responses)

**Prompt template**:
```
Read the following text and answer the question with 0 (no) or 1 (yes).

TEXT:
{trajectory_text}

QUESTION: {question}

Answer with just 0 or 1:
```

### 2. Graded Judgements

Scale questions scored continuously from 0.0 to 1.0.

```json
{
  "graded_judgements": [
    "How masculine is the protagonist?",
    "How violent is this text?"
  ]
}
```

**Prompt template**:
```
Read the following text and answer the question with a score between 0.0 and 1.0.
0.0 means completely no/false, 1.0 means completely yes/true.

TEXT:
{trajectory_text}

QUESTION: {question}

Answer with just a number between 0.0 and 1.0:
```

### 3. Similarity Scoring

Embedding-based cosine similarity to reference words/phrases.

```json
{
  "similarity_scoring": [
    "feminine",
    ["gay", "queer", "LGBTQ+"]
  ]
}
```

**How it works**:
1. Embed the trajectory text using sentence-transformers
2. Embed each reference word/phrase
3. Compute cosine similarity between trajectory and reference embeddings
4. For grouped references, average the similarities

**Default model**: `all-MiniLM-L6-v2`

### 4. Count Occurrences

Lightweight word/phrase frequency scoring without LLM or embeddings.

```json
{
  "count_occurrences": [
    "boy",
    ["cat", "kitten", "feline"]
  ]
}
```

**How it works**:
1. Count occurrences of each word/phrase in the text
2. Normalize by total word count: (# occurrences) / (# total words)
3. For grouped terms, each term is counted separately

**Pros**: Fast, no model required
**Cons**: Only counts exact matches (case-insensitive by default)

## Grouped Structures

Questions can be grouped to average into a single structure:

```json
{
  "categorical_judgements": [
    ["Does this mention a person?", "Does this mention a human?"],
    "Does this mention an animal?"
  ]
}
```

This produces two structures:
- `c1`: Average of the two "person/human" questions
- `c2`: Single "animal" question

Grouping is useful when:
- Multiple formulations capture the same concept
- You want a single aggregate measure for related properties

## String Selection

Control which portion of the trajectory is scored:

| Option | Description |
|--------|-------------|
| `WholeTrajectory` | Full text including prompt |
| `WholeContinuation` | Generated response only (default) |
| `AfterTrunk` | Continuation minus trunk |
| `AfterBranch` | Continuation minus trunk and branch |

```json
{
  "string_selection": "AfterBranch"
}
```

## Output Format

Scoring outputs are saved to `out/<method>/score_<gen>_<scoring>.json`:

```json
{
  "generation_file": "out/simple-sampling/gen_example.json",
  "judge_model": "Qwen/Qwen3-4B-Instruct-2507",
  "categorical_judgements": [
    ["Does this story talk about a person?", "Does this story talk about a boy?"],
    "Does this story talk about someone happy?"
  ],
  "graded_judgements": [...],
  "similarity_scoring": [...],
  "branches": ["trunk", "branch_1", "branch_2"],
  "results": [
    {
      "traj_idx": 0,
      "branch": "trunk",
      "branch_idx": 0,
      "text": "...",
      "scores": [1, 1, 0],
      "graded_scores": [0.7, 0.3],
      "similarity_scores": [0.5, 0.6],
      "conditional_logprobs": {"trunk": -12.5, "branch_1": 0.0},
      "n_continuation_tokens": 45
    }
  ]
}
```

## Conditional Log-Probabilities

Each trajectory stores conditional log-probabilities for each group:

```json
{
  "conditional_logprobs": {
    "trunk": -12.5,
    "branch_1": -8.2,
    "branch_2": 0.0
  }
}
```

- `p(continuation | trunk)`: Probability of the continuation given just the trunk
- `p(continuation | branch_N)`: Probability given the specific branch prefix

These are used in estimation for probability-weighted cores.

## Practical Considerations

### Judge Model Selection

- **Larger is better**: Bigger models give more reliable judgments
- **Instruction-tuned**: Required for following judgment format
- **Same vs. different model**: Using the same model as generation may introduce biases

### Question Design

- **Be specific**: "Does this mention women?" is clearer than "Is this about women?"
- **Avoid ambiguity**: Define terms if needed ("explicitly mention" vs. "imply")
- **Test questions**: Run on sample texts to verify they work as expected

### Handling Edge Cases

- **Thinking tags**: Models with reasoning (Qwen3) emit `<think>...</think>` tags
- **Verbose responses**: Parse for 0/1 even in longer answers
- **Parse failures**: Return `None` (treated as 0.5 in estimation)

### Embedding Model Selection

- Default `all-MiniLM-L6-v2` is fast and general-purpose
- Larger models may capture more nuance
- Domain-specific models may work better for specialized tasks
