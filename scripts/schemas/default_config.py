"""Default configuration values for all experiments.

Centralized location for all default parameter values.
Modify these to change defaults across all scripts.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Generation Defaults
# ══════════════════════════════════════════════════════════════════════════════

# General
TEMPERATURE = 1.0
MAX_NEW_TOKENS = 128

# Simple sampling
SAMPLING_SAMPLES_PER_BRANCH = 10

# Forking paths
FORKING_MAX_ALTERNATES = 5
FORKING_MIN_PROB = 0.2
FORKING_MIN_ENTROPY = 1.75
FORKING_SAMPLES_PER_FORK = 3

# Entropy seeking
ENTROPY_SAMPLES_PER_EXPANSION = 2
ENTROPY_NUM_EXPANSION_ROUNDS = 3

# ══════════════════════════════════════════════════════════════════════════════
# Scoring/Judgment Defaults
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_MAX_TOKENS = 10

# ══════════════════════════════════════════════════════════════════════════════
# Embedding Defaults
# ══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
