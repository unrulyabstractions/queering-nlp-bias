# src/viz/

Tree visualization and plotting tools.

See the main [README.md](../../README.md) for an overview of the project.

## Directory Structure

```
viz/
├── plot.py    # Matplotlib-based tree visualization
└── trees.py   # Tree data structures for rendering
```

## Visualization

### visualize_experiment

Main entry point for rendering experiment results:

```python
from src.viz import visualize_experiment

# Render tree from experiment outputs
visualize_experiment(
    gen_output=generation_output,
    judge_output=judgment_output,
    est_output=estimation_output,
    out_path="out/tree.png",
)
```

### plot_tree

Lower-level tree rendering:

```python
from src.viz.plot import plot_tree

plot_tree(
    tree=token_tree,
    runner=model_runner,
    mode="phrase",           # "token", "word", or "phrase"
    show_probs=True,
    structure_scores=scores,  # Optional: scores to display on leaves
    out_path="tree.png",
)
```

## Tree Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `token` | BPE token-level tree | Detailed token analysis |
| `word` | Whitespace-split words | Readable text |
| `phrase` | Collapsed single-child chains | Compact visualization |

### Token Mode

Shows every BPE token as a separate node:

```
"The" → "▁boy" → "▁went" → "▁to" → "▁the" → "▁park"
```

### Word Mode

Groups tokens into words:

```
"The" → "boy" → "went" → "to" → "the" → "park"
```

Word probabilities are computed as the product of constituent token probabilities.

### Phrase Mode (Default)

Collapses chains where there's only one continuation:

```
"The boy went" → "to the park"
                → "home"
```

## Tree Nodes

### TreeNode

Data structure for visualization tree nodes:

```python
from src.viz.trees import TreeNode

node = TreeNode(
    text="boy",
    token_ids=[123],
    logprob=-0.5,
    prob=0.607,
    children=[...],
    scores={"animal": 0, "human": 1},
)
```

### build_tree

Constructs visualization tree from trajectories:

```python
from src.viz.trees import build_tree

tree_node = build_tree(
    trajectories=trajectories,
    mode="phrase",
)
```

## Styling

The visualization includes:

- **Prompt text**: Gray, italic
- **Template parts**: Light gray
- **Continuations**: Black, bold
- **Probability annotations**: On edges, showing P(token)
- **Structure scores**: On leaf nodes, showing judgment results

## Example Output

```python
from src.inference import ModelRunner
from src.common.token_tree import TokenTree
from src.viz import visualize_experiment

# Load data
runner = ModelRunner("Qwen/Qwen3-0.6B")
gen_output = GenerationOutput.load("out/gen_test.json")

# Generate visualization
visualize_experiment(
    gen_output=gen_output,
    out_path="out/tree_test.png",
)
```

Output: A tree diagram showing:
- Branching structure of generated trajectories
- Probability of each branch
- Optional structure scores on leaf nodes
