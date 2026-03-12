# Analysis Module

Token tree analysis for computing metrics on trajectories, forks, and nodes.

## Overview

Provides analysis infrastructure for token trees with two main components:

1. **Analysis computation** - Main entry point (`analyze_token_tree`) that mutates tree in place
2. **Serialization support** - `DistributionalAnalysis` base class automatically converts logprobs to probabilities during serialization

## Modules

- `tree_analysis.py` - Entry point `analyze_token_tree()` that analyzes trajectories, forks, and nodes
- `analysis_builders.py` - Builder functions: `build_fork_analysis()`, `build_node_analysis()`
- `distributional_analysis_base.py` - `DistributionalAnalysis` base class for analysis dataclasses
- `metrics/` - Structured dataclasses for fork/node/trajectory analysis results

## Usage

```python
from src.common.analysis import analyze_token_tree

# Analyze a token tree (mutates in place)
analyze_token_tree(tree)

# Access results
for traj in tree.trajs:
    print(traj.analysis.metrics.perplexity)

for fork in tree.forks:
    print(fork.analysis.metrics.log_odds)

for node in tree.nodes:
    print(node.analysis.metrics.vocab_entropy)
```

## DistributionalAnalysis

Base class that automatically expands logprob fields to probability fields:
- `*_logprob` → `*_prob` (via exp)
- `*_logprobs` → `*_probs` (via exp)
- `log_odds` → `odds` (via exp)

Used by all analysis dataclasses to provide both logprob and probability representations in serialized output.
