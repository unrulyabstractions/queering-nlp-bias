# Analysis Module

> **Note**: This documentation was AI-generated and may contain errors. If something seems off, check the code or open an issue.


Token tree analysis for computing metrics on trajectories, forks, and nodes.

## Contents

- `analyze.py` - Main entry point (`analyze_token_tree`)
- `base.py` - `DistributionalAnalysis` base class (auto-converts logprobs to probs)
- `builders.py` - Builder functions for analysis objects
- `metrics/` - Analysis dataclasses for forks, nodes, trajectories

## Usage

```python
from src.common.analysis import analyze_token_tree

# Analyze a token tree (mutates in place)
analyze_token_tree(tree)

# Access results
for traj in tree.trajs:
    print(traj.analysis.perplexity)

for fork in tree.forks:
    print(fork.analysis.log_odds)
```

## DistributionalAnalysis

Base class that automatically expands logprob fields to probability fields during serialization:

```python
@dataclass
class MyAnalysis(DistributionalAnalysis):
    token_logprob: float  # In to_dict(), adds token_prob = exp(token_logprob)
```
