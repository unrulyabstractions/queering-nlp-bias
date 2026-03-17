"""Core shared components: config, types, LLM clients, and shared UI."""

from webapp.common.algorithm_config import AlgorithmEvent, SamplingConfig
from webapp.common.llm_clients import (
    JudgeResult,
    generate_from_llm,
    get_client,
    judge_all_questions,
    llm_judge,
    multi_judge,
    multi_judge_all_questions,
)
from webapp.common.normativity_types import (
    GenerationNode,
    NormativityEstimate,
    Scoring,
    Structure,
    System,
    compute_deviation,
    compute_l2_distance,
    compute_l2_norm,
    compute_mean,
    compute_system_means,
    get_word_positions,
    parse_judge_score,
)
from webapp.common.sampling_loop import (
    SamplingState,
    build_state,
    run_sampling_loop,
    serialize_node,
    serialize_state,
)
