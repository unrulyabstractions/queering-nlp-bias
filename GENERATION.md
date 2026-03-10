# Generation Methodology

How we sample trajectories from the LLM's distribution.

## Conceptual Framework

An LLM defines a **tree of strings**: the root is the start token, each node is a string, leaves are complete trajectories, and edges connect strings by next-token continuations with probability `p(t_{p+1}|x_p)`.

Our goal is to sample from this tree to approximate the distribution of outputs, then analyze patterns in what gets generated.

## Branching Structure

Each experiment has:

- **Prompt**: The instruction/context provided to the model
- **Trunk**: Shared prefix that all trajectories start with
- **Branches**: Alternative prefixes that create comparison groups

```
Prompt: "Write a story..."
         │
         └─ Trunk: "Once upon a time, there was a"
                    │
                    ├─ Branch A: " boy" → [trajectories...]
                    └─ Branch B: " cat" → [trajectories...]
```

This creates natural comparison groups: we can compare how stories about boys vs. cats unfold.

## Generation Methods

### 1. Simple Sampling

Sample trajectories using temperature-scaled multinomial sampling.

```bash
python scripts/generate_by_simple_sampling.py trials/generation/example.json \
    --samples-per-branch 10
```

**How it works**:
1. For each branch, prefill with trunk + branch tokens
2. Sample `n` complete trajectories using temperature sampling
3. Record token IDs, log-probabilities, and decoded text

**Parameters**:
- `--samples-per-branch N`: Number of trajectories per branch (default: 2)

**Pros**: Simple, unbiased samples from the distribution
**Cons**: May miss rare modes; many samples needed for coverage

### 2. Forking Paths

Systematically explore one-step deviations from the greedy path.

```bash
python scripts/generate_by_forking_paths.py trials/generation/example.json \
    --max-alternates-per-position 5 \
    --min-prob-for-alternate 0.01 \
    --min-entropy-to-fork 1.0 \
    --samples-per-fork 2
```

**How it works**:
1. Generate the greedy path (always pick highest-probability token)
2. At each position, identify alternate tokens meeting probability threshold
3. For each alternate, spawn continuations from that deviation point
4. Record the resulting tree structure

**Parameters**:
- `--max-alternates-per-position K`: Max alternates to consider per position (default: 3)
- `--min-prob-for-alternate P`: Minimum probability for an alternate (default: 0.05)
- `--min-entropy-to-fork H`: Minimum entropy at position to consider forking (default: 0.0)
- `--samples-per-fork N`: Continuations to sample per fork point (default: 1)

**Pros**: Systematic coverage of near-alternatives; reveals what the model "almost" said
**Cons**: May miss alternatives far from the greedy path

### 3. Seeking Entropy

Expand the tree at high-uncertainty positions.

```bash
python scripts/generate_by_seeking_entropy.py trials/generation/example.json \
    --samples-per-expansion 3 \
    --num-expansion-rounds 4
```

**How it works**:
1. Generate initial trajectories
2. Identify positions with highest entropy (most uncertainty)
3. Expand the tree at those positions by sampling alternatives
4. Repeat for multiple rounds

**Parameters**:
- `--samples-per-expansion N`: Trajectories to sample per expansion (default: 2)
- `--num-expansion-rounds K`: Number of expansion rounds (default: 3)

**Pros**: Focuses computational effort on uncertain decisions
**Cons**: May over-explore noisy positions; entropy isn't always meaningful

## Output Format

Generation outputs are saved to `out/gen_<method>_<config>.json`:

```json
{
  "config": {
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "...",
    "trunk": "...",
    "branches": ["trunk", " boy", " cat"],
    "temperature": 1.0,
    "max_new_tokens": 128
  },
  "method": "sampling",
  "num_trajectories": 30,
  "tree": {
    "trajs": [
      {
        "idx": 0,
        "token_ids": [1, 2, 3, ...],
        "logprobs": [-0.5, -1.2, ...],
        "continuation_text": "...",
        "group_idx": [0]
      }
    ],
    "trunk_length": 15,
    "prompt_length": 10,
    "forks": [...]
  },
  "eos_token": "<|im_end|>"
}
```

## Key Data Structures

### TokenTree

The central data structure organizing multiple trajectories:

```python
tree = TokenTree.from_trajectories(
    trajs=[traj1, traj2, ...],
    groups_per_traj=[(0,), (0,), (1,), ...],
    fork_arms=["boy", "girl"],
    trunk=[0, 1, 2, ...],  # Shared prefix positions
)
```

### TokenTrajectory

Individual token sequence with log-probabilities:

```python
traj = TokenTrajectory(
    token_ids=[1, 2, 3, 4],
    logprobs=[0.0, -0.5, -1.2, -0.3],
)
```

## Practical Considerations

### Model Selection

- Smaller models (0.6B-4B) are faster for experimentation
- Instruction-tuned models produce more coherent text but may be more normative
- Base models may show more raw distributional patterns

### Temperature

- `temperature=1.0`: Sample from actual distribution
- `temperature<1.0`: Concentrate on high-probability paths (more normative)
- `temperature>1.0`: Flatten distribution (more diverse but less coherent)

### Token Limits

- `max_new_tokens` controls trajectory length
- Shorter trajectories are faster but may not develop full patterns
- Longer trajectories provide more signal but cost more

### Branching Design

- **Minimal pairs**: Single-token differences (boy/girl) isolate specific effects
- **Semantic clusters**: Multiple related tokens explore a concept
- **Empty branches**: Trunk-only sampling shows baseline behavior
