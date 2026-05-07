"""Shared helpers for re-scoring arms with new judgments.

Used by `reprocess_arm.py` (read-only inspection) and
`integrate_new_judgments.py` (in-place data integration).
"""

from __future__ import annotations

import json
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.common.callback_types import LogFn
from src.common.math.logit_kde import logit_kde_mode
from src.generation.greedy_output import GreedyOutput
from src.scoring.scorer import Scorer
from src.scoring.scoring_config import ScoringConfig, StringSelection

PARALLEL_WORKERS = 8


# ─── Config / IO ─────────────────────────────────────────────────────────────


def build_scorer(
    *,
    judge_model: str,
    categorical: list[str],
    graded: list[str],
) -> Scorer:
    """Build a Scorer that emits exactly the *new* questions in canonical order:
    all categoricals first, then all gradeds.
    """
    scoring_data: dict[str, list[str | list[str]]] = {}
    if categorical:
        scoring_data["categorical_judgements"] = list(categorical)
    if graded:
        scoring_data["graded_judgements"] = list(graded)
    config = ScoringConfig(
        model=judge_model,
        string_selection=StringSelection.NonThinkingContinuation,
        scoring_data=scoring_data,
    )
    return Scorer(config)


def load_scoring(scoring_path: Path) -> dict[str, Any]:
    with open(scoring_path, encoding="utf-8") as f:
        return json.load(f)


def load_greedy_alongside(scoring_path: Path) -> tuple[Path | None, GreedyOutput | None]:
    """Locate `greedy.json` next to the underlying generation.json."""
    scoring = load_scoring(scoring_path)
    gen_file = scoring.get("metadata", {}).get("generation_file") or ""
    if not gen_file:
        return None, None
    p = GreedyOutput.compute_path(gen_file)
    if not p.exists():
        return p, None
    return p, GreedyOutput.load(p)


# ─── Scoring ─────────────────────────────────────────────────────────────────


@dataclass
class ArmRescore:
    """New-question scores for one arm."""

    arm: str
    sample_traj_indices: list[int] = field(default_factory=list)
    sample_scores: list[list[float]] = field(default_factory=list)
    sample_raw: list[list[str]] = field(default_factory=list)
    greedy_scores: list[float] = field(default_factory=list)
    greedy_raw: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.sample_scores)


def _flat_methods(scorer: Scorer) -> list[str]:
    """Return the active method names in the order Scorer flattens scores."""
    return list(scorer.config.get_active_methods())


def _score_text_detailed(
    scorer: Scorer, text: str
) -> tuple[list[float], list[str]]:
    """Return (flat_scores, flat_raw_responses) in canonical (cat..grade..) order."""
    detailed = scorer.score_detailed(text)
    flat_scores: list[float] = []
    flat_raw: list[str] = []
    for method_name in _flat_methods(scorer):
        items = scorer.config.get_method_items(method_name)
        if not items or method_name not in detailed:
            continue
        scores, raw = detailed[method_name]
        for i, item in enumerate(items):
            score = scores[i] if i < len(scores) else None
            raw_resp = raw[i] if i < len(raw) else ""
            if isinstance(item, list):
                # bundled — average sub-scores
                if isinstance(score, list) and score:
                    flat_scores.append(
                        sum(float(s or 0.0) for s in score) / len(score)
                    )
                else:
                    flat_scores.append(0.0)
                flat_raw.append(json.dumps(raw_resp) if isinstance(raw_resp, list) else str(raw_resp))
            else:
                flat_scores.append(float(score or 0.0))
                flat_raw.append(str(raw_resp))
    return flat_scores, flat_raw


def score_arm(
    scorer: Scorer,
    arm: str,
    scoring_data: dict[str, Any],
    greedy: GreedyOutput | None,
    *,
    log_fn: LogFn | None = None,
) -> ArmRescore:
    """Score every trajectory of `arm` (and its greedy entry, if present)."""
    sample_results = [r for r in scoring_data["results"] if r.get("arm") == arm]
    texts = [r.get("text", "") for r in sample_results]
    indices = [r["traj_idx"] for r in sample_results]

    sample_scores: list[list[float] | None] = [None] * len(texts)
    sample_raw: list[list[str] | None] = [None] * len(texts)

    def _one(idx: int) -> tuple[int, list[float], list[str]]:
        s, r = _score_text_detailed(scorer, texts[idx])
        return idx, s, r

    if texts:
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = [ex.submit(_one, i) for i in range(len(texts))]
            done = 0
            for fut in as_completed(futures):
                idx, s, raw = fut.result()
                sample_scores[idx] = s
                sample_raw[idx] = raw
                done += 1
                if log_fn and (done % 50 == 0 or done == len(texts)):
                    log_fn(f"    [{arm}] scored {done}/{len(texts)}")

    out = ArmRescore(
        arm=arm,
        sample_traj_indices=indices,
        sample_scores=[s or [] for s in sample_scores],
        sample_raw=[r or [] for r in sample_raw],
    )

    if greedy is not None:
        entry = greedy.get_arm(arm)
        if entry is not None:
            out.greedy_scores, out.greedy_raw = _score_text_detailed(scorer, entry.text)
            if log_fn:
                log_fn(f"    [{arm}] greedy: {[round(s, 3) for s in out.greedy_scores]}")

    return out


# ─── Per-arm cores ───────────────────────────────────────────────────────────


def compute_cores(rescore: ArmRescore) -> dict[str, list[float]]:
    """Return four per-method cores for `rescore`'s new-question scores."""
    if not rescore.sample_scores or not rescore.sample_scores[0]:
        return {}
    n = len(rescore.sample_scores[0])
    per_q = [[s[i] for s in rescore.sample_scores] for i in range(n)]
    avg = [sum(q) / len(q) if q else 0.0 for q in per_q]
    return {
        "As Average": avg,
        "As Greedy Decoding": list(rescore.greedy_scores) if rescore.greedy_scores else avg,
        "As Mode": [logit_kde_mode(q) for q in per_q],
        "As Median": [float(statistics.median(q)) for q in per_q],
    }


# ─── Multi-arm orchestration ─────────────────────────────────────────────────


def list_arms(scoring_data: dict[str, Any]) -> list[str]:
    """Distinct arm names in the order encountered."""
    seen: list[str] = []
    for r in scoring_data["results"]:
        a = r.get("arm")
        if a and a not in seen:
            seen.append(a)
    return seen


def score_all_arms(
    scorer: Scorer,
    scoring_data: dict[str, Any],
    greedy: GreedyOutput | None,
    *,
    arms: list[str] | None = None,
    log_fn: LogFn | None = None,
) -> dict[str, ArmRescore]:
    arms = arms or list_arms(scoring_data)
    out: dict[str, ArmRescore] = {}
    for arm in arms:
        if log_fn:
            log_fn(f"  arm: {arm}")
        out[arm] = score_arm(scorer, arm, scoring_data, greedy, log_fn=log_fn)
    return out
