"""Zero-shot NLI classification scoring method.

References:
-----------
Yin, W., Hay, J., & Roth, D. (2019).
"Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach"
EMNLP 2019. https://aclanthology.org/D19-1404
(Original zero-shot classification via NLI)

Goldzycher, J., & Schneider, G. (2022).
"Hypothesis Engineering for Zero-Shot Hate Speech Detection"
TRAC @ LREC 2022. https://aclanthology.org/2022.trac-1.10
(Multi-hypothesis scoring improves accuracy 7.9-10.0 pp)

Laurer, M. et al. (2022).
"Less Annotating, More Classifying: Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT-NLI"
(DeBERTa-v3-mnli-fever-anli-ling-wanli model)

Method:
-------
Natural Language Inference (NLI) models determine whether a hypothesis is
entailed by, contradicted by, or neutral to a premise. This method repurposes
NLI for classification:
- Premise = input text
- Hypothesis = target label as natural language statement
- Score = P(entailment)

Multi-hypothesis scoring probes different dimensions of the target concept.
Each hypothesis is scored independently (multi_label=True), then aggregated.

Aggregation options:
- mean: s = (1/K) * sum(p_k)  -- all dimensions weighted equally
- max: s = max(p_k)           -- strongest single signal
- noisy_or: s = 1 - prod(1-p_k)  -- probability any dimension fires

Advantages over LLM-as-judge:
- Free (runs locally), ~50ms per string on GPU
- Deterministic (same input = same output)
- No API required

Limitations:
- No cultural reasoning (pattern matching on entailment)
- Hypothesis-sensitive (exact wording matters)
- 512 token context limit (longer texts need chunking)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from src.common.callback_types import LogFn
from src.inference import ModelRunner
from src.inference.embedding_runner import EmbeddingRunner

from ..scoring_method_registry import ScoringMethodParams, register_method

# ══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

# Default NLI model - strong performance on MNLI
# See: https://huggingface.co/facebook/bart-large-mnli
DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"


@dataclass
class NliParams(ScoringMethodParams):
    """Parameters for zero-shot NLI classification scoring.

    The method scores text against multiple hypotheses using an NLI model.
    Each hypothesis is scored independently (multi_label=True), and scores
    are aggregated via mean, max, or noisy_or.
    """

    nli_model: str = field(default_factory=lambda: DEFAULT_NLI_MODEL)
    aggregation: str = "mean"  # "mean", "max", or "noisy_or"

    # Registry metadata
    name: ClassVar[str] = "nli"
    config_key: ClassVar[str] = "nli_hypotheses"
    label_prefix: ClassVar[str] = "n"
    requires_runner: ClassVar[bool] = False  # Uses own NLI model
    requires_embedder: ClassVar[bool] = False

    _cli_args: ClassVar[dict[str, str]] = {
        "nli_model": "--nli-model",
        "aggregation": "--aggregation",
    }


# ══════════════════════════════════════════════════════════════════════════════
# NLI PIPELINE SINGLETON
# ══════════════════════════════════════════════════════════════════════════════

# Cache the pipeline to avoid reloading for each trajectory
_NLI_PIPELINE: Any = None
_NLI_MODEL_NAME: str = ""


def get_nli_pipeline(model_name: str) -> Any:
    """Get or create NLI pipeline (cached singleton)."""
    global _NLI_PIPELINE, _NLI_MODEL_NAME

    if _NLI_PIPELINE is not None and _NLI_MODEL_NAME == model_name:
        return _NLI_PIPELINE

    from transformers import pipeline

    from src.common.device_utils import get_device
    from src.common.logging import log

    device = get_device()
    # transformers pipeline uses device index for GPU
    device_arg = 0 if device == "cuda" else -1 if device == "cpu" else device

    log(f"Loading NLI model: {model_name} on {device}...")
    _NLI_PIPELINE = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device_arg,
    )
    _NLI_MODEL_NAME = model_name
    log(f"NLI model loaded: {model_name}")

    return _NLI_PIPELINE


def cleanup_nli_pipeline() -> None:
    """Release NLI pipeline memory."""
    global _NLI_PIPELINE, _NLI_MODEL_NAME

    if _NLI_PIPELINE is not None:
        del _NLI_PIPELINE
        _NLI_PIPELINE = None
        _NLI_MODEL_NAME = ""

        from src.common.device_utils import clear_gpu_memory

        clear_gpu_memory()


# ══════════════════════════════════════════════════════════════════════════════
# SCORING LOGIC
# ══════════════════════════════════════════════════════════════════════════════


def score_text_nli(
    text: str,
    hypotheses: list[str],
    nli_pipeline: Any,
    aggregation: str = "mean",
) -> dict[str, Any]:
    """Score text against hypotheses using NLI.

    Args:
        text: Input text (premise)
        hypotheses: List of hypothesis strings
        nli_pipeline: HuggingFace zero-shot-classification pipeline
        aggregation: "mean", "max", or "noisy_or"

    Returns:
        Dict with 'score' (aggregated) and 'per_hypothesis' (individual scores)
    """
    # CRITICAL: multi_label=True scores each hypothesis independently
    # Without this, scores are normalized across hypotheses (mutually exclusive)
    result = nli_pipeline(
        text,
        candidate_labels=hypotheses,
        multi_label=True,
    )

    # result["labels"] = hypotheses sorted by score
    # result["scores"] = corresponding P(entailment) values
    per_hypothesis = dict(zip(result["labels"], result["scores"]))

    # Get scores in original hypothesis order
    scores = [per_hypothesis[h] for h in hypotheses]

    # Aggregate
    if aggregation == "mean":
        final_score = sum(scores) / len(scores) if scores else 0.0
    elif aggregation == "max":
        final_score = max(scores) if scores else 0.0
    elif aggregation == "noisy_or":
        # P(at least one hypothesis entailed)
        product = 1.0
        for p in scores:
            product *= 1.0 - p
        final_score = 1.0 - product
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return {
        "score": final_score,
        "per_hypothesis": per_hypothesis,
        "scores_ordered": scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# REGISTERED SCORING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════


@register_method(NliParams)
def score_nli(
    text: str,
    items: list[str | list[str]],
    params: NliParams,
    runner: ModelRunner | None = None,
    embedder: EmbeddingRunner | None = None,
    log_fn: LogFn | None = None,
) -> tuple[list[float], list[str]]:
    """Score text using zero-shot NLI classification.

    Each item in the config is treated as a hypothesis. The NLI model
    computes P(entailment | premise=text, hypothesis=item) for each.
    Scores are aggregated via mean, max, or noisy_or.

    Args:
        text: Text to score (used as NLI premise)
        items: Hypotheses from config (e.g., "This text contains queer subtext.")
        params: Method parameters including nli_model and aggregation
        runner: Not used (method has its own NLI model)
        embedder: Not used
        log_fn: Optional logging callback

    Returns:
        Tuple of ([aggregated_score], [raw_response_summary])
    """
    if not items:
        return [0.0], ["no_hypotheses"]

    # Flatten items (handle bundled hypotheses)
    hypotheses: list[str] = []
    for item in items:
        if isinstance(item, list):
            hypotheses.extend(item)
        else:
            hypotheses.append(item)

    # Filter out placeholder items
    hypotheses = [h for h in hypotheses if h and h != "_"]

    if not hypotheses:
        if log_fn:
            log_fn("No valid hypotheses provided")
        return [0.0], ["no_hypotheses"]

    # Get NLI pipeline
    nli_pipeline = get_nli_pipeline(params.nli_model)

    if log_fn:
        log_fn(f"Scoring with {len(hypotheses)} hypotheses ({params.aggregation})")

    # Score
    result = score_text_nli(text, hypotheses, nli_pipeline, params.aggregation)

    if log_fn:
        log_fn(f"Score: {result['score']:.3f}")
        # Log top hypotheses
        sorted_hyps = sorted(
            result["per_hypothesis"].items(), key=lambda x: x[1], reverse=True
        )
        for hyp, score in sorted_hyps[:3]:
            hyp_preview = hyp[:50] + "..." if len(hyp) > 50 else hyp
            log_fn(f"  {score:.2f}  {hyp_preview}")

    # Build raw response summary
    top_hyps = sorted(
        result["per_hypothesis"].items(), key=lambda x: x[1], reverse=True
    )[:3]
    raw_parts = [f"{h[:30]}={s:.2f}" for h, s in top_hyps]
    raw_summary = "; ".join(raw_parts)

    return [result["score"]], [raw_summary]
