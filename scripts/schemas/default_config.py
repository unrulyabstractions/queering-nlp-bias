"""Default configuration values for all experiments.

Centralized location for all default parameter values.
Modify these to change defaults across all scripts.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Generation Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Sampling temperature (0=greedy, 1=standard, >1=more random)
TEMPERATURE = 1.0

# Max tokens to generate per trajectory
MAX_NEW_TOKENS = 128

# Simple sampling: trajectories per branch (trunk + each branch)
SAMPLING_SAMPLES_PER_BRANCH = 10

# Forking paths: max alternate tokens to consider at each position
FORKING_MAX_ALTERNATES = 5

# Forking paths: min probability for a token to qualify as alternate
FORKING_MIN_PROB = 0.2

# Forking paths: min entropy (nats) at position to consider forking
FORKING_MIN_ENTROPY = 1.75

# Forking paths: continuations to sample from each fork point
FORKING_SAMPLES_PER_FORK = 3

# Entropy seeking: trajectories per expansion round
ENTROPY_SAMPLES_PER_EXPANSION = 2

# Entropy seeking: number of rounds to expand high-entropy positions
ENTROPY_NUM_EXPANSION_ROUNDS = 3

# ══════════════════════════════════════════════════════════════════════════════
# Scoring/Judgment Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Max tokens for judge model response (0/1 answer)
JUDGE_MAX_TOKENS = 10

# Which text to score. Options:
#   "WholeTrajectory"   - Full text including prompt and response
#   "WholeContinuation" - Just the generated continuation (includes trunk and branch tokens)
#   "AfterTrunk"        - Continuation after trunk prefix (includes branch tokens)
#   "AfterBranch"       - Continuation after branch point (excludes branch tokens)
STRING_SELECTION = "WholeContinuation"

# ══════════════════════════════════════════════════════════════════════════════
# Embedding Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Sentence transformer model for similarity scoring
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
