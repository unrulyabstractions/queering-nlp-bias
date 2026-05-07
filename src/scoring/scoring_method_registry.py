"""Scoring method registry with auto-discovery.

Adding a new scoring method is ONE FILE - no other changes needed:

    @dataclass
    class MyParams(ScoringMethodParams):
        name: ClassVar[str] = "my-method"
        config_key: ClassVar[str] = "my_method_data"  # JSON field name
        label_prefix: ClassVar[str] = "m"  # Structure label (m1, m2, ...)

    @register_method(MyParams)
    def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
        # items = config.get_method_items("my-method") - your method's data
        ...

The method is automatically:
- Discovered by the pipeline
- Has its config field read
- Has its scores stored in results
- Has its summary displayed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, TypeVar

from src.common.callback_types import LogFn
from src.common.params_schema import ParamsSchema

if TYPE_CHECKING:
    from src.inference import ModelRunner
    from src.inference.embedding_runner import EmbeddingRunner


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER BASE CLASS
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class ScoringMethodParams(ParamsSchema):
    """Base class for scoring method parameters.

    Subclasses MUST define as ClassVar:
    - name: str - method name for registry lookup
    - config_key: str - JSON field name in scoring config (e.g., "categorical_judgements")
    - label_prefix: str - single letter for structure labels (e.g., "c" -> c1, c2, ...)

    Optionally define:
    - requires_runner: bool - True if method needs LLM (default: False)
    - requires_embedder: bool - True if method needs embeddings (default: False)
    """

    # Subclasses override these as ClassVar
    name: ClassVar[str]
    config_key: ClassVar[str]
    label_prefix: ClassVar[str]

    # Resource requirements (override if needed)
    requires_runner: ClassVar[bool] = False
    requires_embedder: ClassVar[bool] = False


P = TypeVar("P", bound=ScoringMethodParams)

# Type alias for score functions
# Args: (text, items, params, runner, embedder, log_fn)
# Returns: (scores, raw_responses)
ScoreFn = Callable[
    [
        str,
        list[str | list[str]],
        P,
        "ModelRunner | None",
        "EmbeddingRunner | None",
        LogFn | None,
    ],
    tuple[list[Any], list[str]],
]


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

# Registry stores (params_class, score_fn) pairs
_REGISTRY: dict[str, tuple[type[ScoringMethodParams], ScoreFn]] = {}


def register_method(params_class: type[ScoringMethodParams]):
    """Decorator to register a scoring method.

    Usage:
        @dataclass
        class MyParams(ScoringMethodParams):
            name: ClassVar[str] = "my-method"
            config_key: ClassVar[str] = "my_method_data"
            label_prefix: ClassVar[str] = "m"

        @register_method(MyParams)
        def score_my_method(text, items, params, runner=None, embedder=None, log_fn=None):
            # items is the list from config[config_key]
            ...
    """

    def decorator(fn: ScoreFn) -> ScoreFn:
        _REGISTRY[params_class.name] = (params_class, fn)
        return fn

    return decorator


def get_method(name: str) -> ScoreFn:
    """Get a scoring function by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown scoring method: {name}. Valid: {valid}")
    return _REGISTRY[name][1]


def get_default_params(name: str) -> ScoringMethodParams:
    """Get default params for a method (all defaults applied)."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown scoring method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]()


def get_params_class(name: str) -> type[ScoringMethodParams]:
    """Get the params class for a method by name."""
    if name not in _REGISTRY:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown scoring method: {name}. Valid: {valid}")
    return _REGISTRY[name][0]


def list_methods() -> list[str]:
    """List all registered method names."""
    return sorted(_REGISTRY.keys())


def iter_methods() -> list[tuple[str, type[ScoringMethodParams], ScoreFn]]:
    """Iterate over all registered methods.

    Returns:
        List of (name, params_class, score_fn) tuples
    """
    return [(name, pc, fn) for name, (pc, fn) in sorted(_REGISTRY.items())]


def params_from_dict(method: str, data: dict) -> ScoringMethodParams:
    """Create a params instance from a dict using the correct class."""
    params_class = get_params_class(method)
    return params_class.from_dict(data)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════


def safe_max_parallel(runner: Any | None) -> int:
    """Return a safe max-parallel count for question-level scoring.

    Local model backends (MLX, HuggingFace) are NOT thread-safe — concurrent
    `generate()` calls on the same model instance segfault. Only allow
    intra-trajectory parallelism for API-based backends.
    """
    if runner is None:
        return 1
    from src.inference.backends.model_backend import ModelBackend
    api_backends = {ModelBackend.OPENAI, ModelBackend.ANTHROPIC, ModelBackend.GEMINI}
    return 16 if getattr(runner, "backend", None) in api_backends else 1


def score_with_bundling(
    items: list[str | list[str]],
    score_fn: Callable[[str], tuple[Any, str]],
    label_prefix: str,
    log_fn: LogFn | None = None,
    *,
    max_parallel: int = 1,
) -> tuple[list[Any], list[str | list[str]]]:
    """Shared loop for bundled item scoring with intra-trajectory parallelism.

    Handles both single items and bundled items (lists of items).
    Preserves structure: bundled items return nested lists.

    All sub-questions for one trajectory are scored in parallel via a
    bounded thread pool — CoT calls are I/O bound and sequential
    per-question latency dominates total scoring time. Outputs are kept
    in their original positional order.

    Args:
        items: List of items (can be strings or lists of strings for bundles)
        score_fn: Function(item) -> (score, raw_response) to score a single item
        label_prefix: Prefix for logging (e.g., "c", "g", "s")
        log_fn: Optional logging callback
        max_parallel: Max in-flight per-question calls per trajectory.

    Returns:
        Tuple of (scores, raw_responses) preserving bundle structure
    """
    from concurrent.futures import ThreadPoolExecutor

    # Flatten (struct_idx, sub_idx_or_None, item) so we can submit them all
    # to a single pool and stitch results back into the original shape.
    flat: list[tuple[int, int | None, str]] = []
    for struct_idx, item in enumerate(items):
        if isinstance(item, list):
            for sub_idx, sub_item in enumerate(item):
                flat.append((struct_idx, sub_idx, sub_item))
        else:
            flat.append((struct_idx, None, item))

    scores: list[Any] = [None] * len(items)
    raws: list[Any] = [None] * len(items)
    # Pre-fill bundle slots so we can index into them.
    for i, item in enumerate(items):
        if isinstance(item, list):
            scores[i] = [None] * len(item)
            raws[i] = [None] * len(item)

    def _do(task: tuple[int, int | None, str]):
        struct_idx, sub_idx, text = task
        score, raw = score_fn(text)
        return struct_idx, sub_idx, score, raw

    if not flat:
        return scores, raws

    workers = max(1, min(max_parallel, len(flat)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for struct_idx, sub_idx, score, raw in ex.map(_do, flat):
            if sub_idx is None:
                scores[struct_idx] = score
                raws[struct_idx] = raw
            else:
                scores[struct_idx][sub_idx] = score
                raws[struct_idx][sub_idx] = raw

    if log_fn:
        for struct_idx, item in enumerate(items):
            if isinstance(item, list):
                log_fn(f"[{label_prefix}{struct_idx + 1}] Bundled ({len(item)} items)")
                for sub_idx, sub_item in enumerate(item):
                    _log_score(log_fn, sub_item, scores[struct_idx][sub_idx], indent=True)
            else:
                _log_score(
                    log_fn, item, scores[struct_idx],
                    prefix=f"[{label_prefix}{struct_idx + 1}]",
                )

    return scores, raws


def _log_score(
    log_fn: LogFn,
    item: str,
    score: Any,
    indent: bool = False,
    prefix: str = "",
) -> None:
    """Log a single score result."""
    if score is None:
        score_str = "?"
    elif isinstance(score, float):
        score_str = f"{score:.3f}"
    else:
        score_str = str(score)

    item_preview = item[:40] + "..." if len(item) > 40 else item
    if indent:
        log_fn(f"     • {item_preview} -> {score_str}")
    else:
        log_fn(f"{prefix} {item_preview} -> {score_str}")
