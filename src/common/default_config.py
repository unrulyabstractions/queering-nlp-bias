"""Default configuration values for all experiments.

This is the SINGLE SOURCE OF TRUTH for all default parameter values.
Params classes import their defaults from here.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Generation Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Sampling temperature (0=greedy, 1=standard, >1=more random)
TEMPERATURE = 1.0

# Max tokens to generate per trajectory
MAX_NEW_TOKENS = 258

# Simple sampling: trajectories per arm (trunk + each branch)
SAMPLING_SAMPLES_PER_ARM = 10

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

# Which text to score
STRING_SELECTION = "NonThinkingContinuation"


# ══════════════════════════════════════════════════════════════════════════════
# Embedding Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Sentence transformer model for similarity scoring
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# Estimation Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Default statistic for core estimation
DEFAULT_STATISTIC = "standard"

# Default weighting method for estimation (prob = probability-weighted)
DEFAULT_WEIGHTING_METHOD = "prob"


# ══════════════════════════════════════════════════════════════════════════════
# Dynamics Defaults
# ══════════════════════════════════════════════════════════════════════════════

# Measure scores every N tokens
DYNAMICS_STEP = 4

# Number of most extremal trajectories to analyze per arm
DYNAMICS_TRAJS_PER_ARM = 2

# Which arm types to analyze: "root", "trunk", "branch", "twig"
DYNAMICS_ARMS: list[str] = ["branch"]
