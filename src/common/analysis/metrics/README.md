# Analysis Metrics

Dataclasses for analyzing token trajectories, branching nodes, and binary forks in token trees.

## Overview

This module provides structured metrics for three levels of analysis:
1. **Trajectories**: Sequence-level metrics (perplexity, cross-entropy, ranks)
2. **Branching nodes**: Vocabulary distribution metrics at decision points
3. **Binary forks**: Comparative metrics for two competing token choices

All metrics inherit from `DistributionalAnalysis` base class and support serialization via `BaseSchema`.

## Modules

### trajectory_analysis_types.py

Core classes:
- **`TrajectoryMetrics`**: Metrics computed over a trajectory's logprob sequence
  - Distributional: `empirical_cross_entropy`, `inv_perplexity`, `perplexity`, `total_logprob`
  - Worst-token tracking: `worst_token_logprob`, `worst_token_position`
  - Rank-based (optional): `worst_token_rank`, `worst_rank_position`
  - Top-p normalized (optional): `top_p_normalized` (TopPNormalizedMetrics)
  - Methods: `from_logprobs()`, `from_trajectory()`

- **`TopPNormalizedMetrics`**: Top-p normalized probability metrics
  - `p`: Number of top tokens considered
  - `total_logprob`: Sum of normalized logprobs
  - `worst_token_logprob`: Minimum normalized logprob
  - `worst_token_position`: Position of worst token
  - Method: `from_logits()` - builds from full logits for a token slice

- **`TrajectoryAnalysis`**: Container for trajectory metrics with trunk/continuation breakdown
  - `full_traj`: Metrics over entire trajectory
  - `trunk_only`: Metrics over prompt (if trunk_length provided)
  - `continuation_only`: Metrics over generated portion (if applicable)
  - Methods: `from_trajectory()`, `from_logprobs()`

### fork_analysis_types.py

Core classes:
- **`ForkMetrics`**: Metrics for binary fork (two competing tokens)
  - Entropy-based: `fork_entropy`, `fork_diversity` (D₁), `fork_simpson` (D₂), `fork_concentration`
  - Probability comparison: `probability_ratio`, `log_odds`, `logit_diff`, `reciprocal_rank_a`
  - Raw logprobs: `next_token_logprobs` (tuple of two values)

- **`ForkAnalysis`**: Wrapper for fork metrics at a specific index
  - `fork_idx`: Position of fork in tree
  - `metrics`: ForkMetrics instance

### node_metrics.py

Core classes:
- **`NodeMetrics`**: Metrics at a vocabulary distribution decision point
  - `vocab_entropy`: Full vocabulary entropy at branching node
  - `vocab_diversity`: Effective vocabulary size (e^entropy)
  - `next_token_logprobs`: Logprobs of candidate tokens

- **`NodeAnalysis`**: Wrapper for node metrics at a specific index
  - `node_idx`: Position of branching node
  - `metrics`: NodeMetrics instance
