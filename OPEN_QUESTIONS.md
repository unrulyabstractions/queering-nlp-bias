# Open Questions

> **Note**: This documentation was LLM-generated. If something seems wrong or contradicts the code, please report bugs.

Research questions organized by pipeline stage.

## Generation

- **Sampling strategies**: Which generation method best captures the diversity of the model's distribution?
  - Simple sampling explores randomly but may miss rare modes
  - Forking paths is systematic but may not reach distant alternatives
  - Entropy-seeking prioritizes uncertainty but may over-explore noise

- **Coverage vs. efficiency**: How many trajectories are needed to estimate the core reliably? What's the trade-off between computational cost and estimate quality?

- **Temperature effects**: Does temperature increase meaningful diversity or just incoherence? Can we disentangle these?

- **Prompt design**: How does prompt structure affect the distribution?
  - System prompts vs. user prompts vs. few-shot examples
  - Explicit instructions ("be diverse") vs. implicit framing
  - How much context shapes what's considered "normative"?

- **Branching design**: How should branches be chosen to reveal bias?
  - Minimal pairs (boy/girl) vs. semantic clusters (queer/straight/non-binary)
  - Token-level vs. phrase-level branching
  - How to avoid confirmation bias in branch selection?


## Scoring

- **Scoring specification**: How should questions be designed to capture meaningful dimensions?
  - What makes a "good" categorical question?
  - When to group vs. keep separate?
  - How to avoid leading questions?

- **Judge reliability**: How consistent are LLM judges? Do different judges reveal different patterns?

- **Graded vs. binary**: When should we use graded (0-1) vs. binary (yes/no) judgments? What information is lost/gained?

- **Similarity scoring**: What embedding models work best for semantic similarity? How to choose reference words/phrases?

- **Cross-scoring dependencies**: Some structures may be correlated (e.g., "mentions man" and "masculine protagonist"). How should this affect analysis?

- **Prompt engineering for judges**: How much does the judgment prompt affect results? How to make judgments robust?


## Estimation

- **Selection of statistics**: The (q, r) generalized core offers many options:
  - q: power mean order (arithmetic, geometric, harmonic, max, min)
  - r: escort order (actual distribution, uniform, mode, anti-mode)
  - Which combinations are most meaningful for which purposes?
  - Different (q, r) answer different questions: "what does the model usually do?" vs. "what does the model do at its most confident?"

- **Weighting schemes**: Probability-weighted vs. inverse-perplexity-weighted cores give different views.
  - Probability-weighted is canonical (matches true distribution)
  - Inverse-perplexity-weighted is tractable (normalizes for sequence length)
  - When do they diverge? What does divergence indicate?

- **Cross-branch comparison**: How to compare cores across branches? What metrics capture meaningful differences?
  - Core distance (L2 between core vectors)?
  - Deviance ratio (E[∂] in branch A vs. branch B)?
  - Statistical tests for significance?
