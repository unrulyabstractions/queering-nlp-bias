"""Tests for whistles, marked_personas, and nli scoring methods."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from src.scoring.methods.whistles_method import (
    GlossaryEntry,
    WhistlesParams,
    aggregate_max,
    aggregate_noisy_or,
    build_whistle_detection_prompt,
    find_glossary_matches,
    load_glossary,
    parse_probability_response,
)
from src.scoring.methods.marked_personas_method import (
    MarkedLexicon,
    MarkedPersonasParams,
    build_word_counts,
    compute_fightin_words_delta,
    score_text_with_lexicon,
    tokenize_simple,
)
from src.scoring.methods.nli_method import (
    NliParams,
    score_text_nli,
)
from src.scoring.scoring_method_registry import get_params_class, list_methods


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRY TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestMethodRegistration:
    """Verify methods are registered in the scoring registry."""

    def test_whistles_registered(self):
        """Whistles method should be registered."""
        assert "whistles" in list_methods()
        params_class = get_params_class("whistles")
        assert params_class.config_key == "whistles"
        assert params_class.label_prefix == "w"
        assert params_class.requires_runner is True

    def test_marked_personas_registered(self):
        """Marked personas method should be registered."""
        assert "marked_personas" in list_methods()
        params_class = get_params_class("marked_personas")
        assert params_class.config_key == "marked_personas"
        assert params_class.label_prefix == "p"
        # Phase 2 (scoring with lexicon) doesn't need runner
        assert params_class.requires_runner is False


# ══════════════════════════════════════════════════════════════════════════════
# WHISTLES TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestGlossaryEntry:
    """Test GlossaryEntry parsing."""

    def test_from_dict_basic(self):
        """Should parse basic glossary entry."""
        entry = GlossaryEntry.from_dict({
            "surface_form": "confirmed bachelor",
            "covert_meaning": "gay man",
            "type": "II",
        })
        assert entry.surface_form == "confirmed bachelor"
        assert entry.covert_meaning == "gay man"
        assert entry.whistle_type == "II"

    def test_from_dict_alternate_field_names(self):
        """Should handle alternate field names like 'term'."""
        entry = GlossaryEntry.from_dict({
            "term": "family values",
            "covert_meaning": "anti-LGBTQ stance",
        })
        assert entry.surface_form == "family values"


class TestGlossaryLoading:
    """Test glossary file loading."""

    def test_load_glossary(self, tmp_path: Path):
        """Should load glossary from JSON file."""
        glossary_data = [
            {"surface_form": "roommates", "covert_meaning": "partners"},
            {"surface_form": "friend", "covert_meaning": "lover"},
        ]
        glossary_file = tmp_path / "glossary.json"
        glossary_file.write_text(json.dumps(glossary_data))

        entries = load_glossary(glossary_file)
        assert len(entries) == 2
        assert entries[0].surface_form == "roommates"
        assert entries[1].covert_meaning == "lover"


class TestGlossaryMatching:
    """Test finding glossary matches in text."""

    def test_find_single_match(self):
        """Should find single matching term."""
        entries = [
            GlossaryEntry("confirmed bachelor", "gay man", "II"),
            GlossaryEntry("roommates", "partners", "II"),
        ]
        matches = find_glossary_matches("He was a confirmed bachelor.", entries)
        assert len(matches) == 1
        assert matches[0].surface_form == "confirmed bachelor"

    def test_find_multiple_matches(self):
        """Should find all matching terms."""
        entries = [
            GlossaryEntry("confirmed bachelor", "gay man", "II"),
            GlossaryEntry("roommates", "partners", "II"),
        ]
        text = "He was a confirmed bachelor living with his roommates."
        matches = find_glossary_matches(text, entries)
        assert len(matches) == 2

    def test_no_matches(self):
        """Should return empty list when no matches."""
        entries = [GlossaryEntry("confirmed bachelor", "gay man", "II")]
        matches = find_glossary_matches("He was an engineer.", entries)
        assert len(matches) == 0

    def test_case_insensitive_matching(self):
        """Should match regardless of case."""
        entries = [GlossaryEntry("Friend of Dorothy", "gay", "I")]
        matches = find_glossary_matches("She is a friend of dorothy.", entries)
        assert len(matches) == 1


class TestProbabilityParsing:
    """Test parsing probability responses from LLM."""

    def test_parse_decimal(self):
        """Should parse decimal probability."""
        assert parse_probability_response("0.85") == 0.85
        assert parse_probability_response("0.5") == 0.5

    def test_parse_with_context(self):
        """Should extract probability from longer response."""
        assert parse_probability_response("The probability is 0.72.") == 0.72

    def test_parse_boundary_values(self):
        """Should handle 0 and 1."""
        assert parse_probability_response("0") == 0.0
        assert parse_probability_response("1") == 1.0
        assert parse_probability_response("1.0") == 1.0

    def test_parse_invalid_returns_none(self):
        """Should return None for unparseable responses."""
        assert parse_probability_response("not a number") is None


class TestAggregation:
    """Test probability aggregation methods."""

    def test_noisy_or_single(self):
        """Noisy-OR with single probability."""
        assert aggregate_noisy_or([0.8]) == 0.8

    def test_noisy_or_multiple(self):
        """Noisy-OR with multiple probabilities."""
        # 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
        assert abs(aggregate_noisy_or([0.5, 0.5]) - 0.75) < 0.001

    def test_noisy_or_empty(self):
        """Noisy-OR with no probabilities."""
        assert aggregate_noisy_or([]) == 0.0

    def test_max_aggregation(self):
        """Max aggregation returns highest."""
        assert aggregate_max([0.3, 0.9, 0.5]) == 0.9

    def test_max_empty(self):
        """Max with no probabilities."""
        assert aggregate_max([]) == 0.0


class TestWhistlePromptBuilding:
    """Test prompt construction for whistle detection."""

    def test_prompt_contains_required_elements(self):
        """Prompt should contain text, term, and covert meaning."""
        entry = GlossaryEntry("bachelor", "gay man", "II")
        demos = [GlossaryEntry("roommates", "partners", "II", example_coded="lived as roommates")]
        prompt = build_whistle_detection_prompt("He was a bachelor.", entry, demos)

        assert "bachelor" in prompt
        assert "gay man" in prompt
        assert "He was a bachelor." in prompt


# ══════════════════════════════════════════════════════════════════════════════
# MARKED PERSONAS TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestTokenization:
    """Test simple word tokenization."""

    def test_basic_tokenization(self):
        """Should split into lowercase words."""
        tokens = tokenize_simple("Hello World")
        assert tokens == ["hello", "world"]

    def test_removes_punctuation(self):
        """Should remove punctuation."""
        tokens = tokenize_simple("Hello, world! How are you?")
        assert tokens == ["hello", "world", "how", "are", "you"]

    def test_removes_numbers(self):
        """Should remove numeric tokens."""
        tokens = tokenize_simple("Chapter 1: Introduction")
        # Only alphabetic words
        assert "1" not in tokens
        assert "chapter" in tokens


class TestWordCounting:
    """Test word frequency counting."""

    def test_count_single_text(self):
        """Should count word frequencies."""
        from collections import Counter
        counts = build_word_counts(["the cat sat on the mat"])
        assert counts["the"] == 2
        assert counts["cat"] == 1

    def test_count_multiple_texts(self):
        """Should aggregate across texts."""
        counts = build_word_counts(["hello world", "hello there"])
        assert counts["hello"] == 2
        assert counts["world"] == 1


class TestMarkedLexicon:
    """Test lexicon save/load."""

    def test_save_and_load(self, tmp_path: Path):
        """Should save and load lexicon correctly."""
        lexicon = MarkedLexicon(
            delta={"resilience": 3.5, "scalable": -2.0},
            marked_label="queer",
            domain="engineer",
            n_samples=100,
        )
        path = tmp_path / "lexicon.json"
        lexicon.save(path)

        loaded = MarkedLexicon.load(path)
        assert loaded.delta == lexicon.delta
        assert loaded.marked_label == "queer"
        assert loaded.n_samples == 100


class TestLexiconScoring:
    """Test text scoring with precomputed lexicon."""

    def test_score_marked_text(self):
        """Text with marked words should have high score."""
        lexicon = MarkedLexicon(
            delta={"resilience": 5.0, "barriers": 4.0},
            marked_label="queer",
            domain="engineer",
            n_samples=100,
        )
        result = score_text_with_lexicon("Showing resilience against barriers.", lexicon)
        assert result["score"] > 0.9  # High marked signal

    def test_score_unmarked_text(self):
        """Text with unmarked words should have low score."""
        lexicon = MarkedLexicon(
            delta={"scalable": -4.0, "infrastructure": -3.0},
            marked_label="queer",
            domain="engineer",
            n_samples=100,
        )
        result = score_text_with_lexicon("Building scalable infrastructure.", lexicon)
        assert result["score"] < 0.1  # Low marked signal

    def test_score_neutral_text(self):
        """Text with no lexicon words should return 0.5-ish."""
        lexicon = MarkedLexicon(
            delta={"resilience": 5.0, "scalable": -5.0},
            marked_label="queer",
            domain="engineer",
            n_samples=100,
        )
        result = score_text_with_lexicon("Hello world.", lexicon)
        # No words in lexicon, denominator is epsilon, numerator is 0
        assert result["score"] == 0.0

    def test_top_words_extraction(self):
        """Should identify top marked and unmarked words."""
        lexicon = MarkedLexicon(
            delta={"resilience": 5.0, "barriers": 4.0, "scalable": -3.0},
            marked_label="queer",
            domain="engineer",
            n_samples=100,
        )
        result = score_text_with_lexicon(
            "Resilience against barriers while building scalable systems.", lexicon
        )
        top_marked = [w for w, _ in result["top_marked"]]
        top_unmarked = [w for w, _ in result["top_unmarked"]]
        assert "resilience" in top_marked
        assert "scalable" in top_unmarked


class TestFightinWordsDelta:
    """Test Fightin' Words z-score computation."""

    def test_positive_delta_for_marked_word(self):
        """Words more frequent in marked should have positive delta."""
        from collections import Counter
        t_marked = Counter({"resilience": 50, "the": 100})
        t_unmarked = Counter({"resilience": 5, "the": 100})
        prior = Counter({"resilience": 10, "the": 1000})

        delta = compute_fightin_words_delta(t_marked, t_unmarked, prior, min_count=3)
        assert delta.get("resilience", 0) > 0

    def test_negative_delta_for_unmarked_word(self):
        """Words more frequent in unmarked should have negative delta."""
        from collections import Counter
        t_marked = Counter({"scalable": 5, "the": 100})
        t_unmarked = Counter({"scalable": 50, "the": 100})
        prior = Counter({"scalable": 10, "the": 1000})

        delta = compute_fightin_words_delta(t_marked, t_unmarked, prior, min_count=3)
        assert delta.get("scalable", 0) < 0


class TestMarkedPersonasParams:
    """Test MarkedPersonasParams defaults and validation."""

    def test_default_values(self):
        """Should have sensible defaults."""
        params = MarkedPersonasParams()
        assert params.n_samples == 100
        assert params.min_word_count == 5
        assert params.marked_label == ""
        assert params.domain == ""

    def test_params_fields(self):
        """Should have all required class vars."""
        assert MarkedPersonasParams.name == "marked_personas"
        assert MarkedPersonasParams.config_key == "marked_personas"
        assert MarkedPersonasParams.label_prefix == "p"


# ══════════════════════════════════════════════════════════════════════════════
# NLI TESTS
# ══════════════════════════════════════════════════════════════════════════════


class TestNliRegistration:
    """Verify NLI method is registered in the scoring registry."""

    def test_nli_registered(self):
        """NLI method should be registered."""
        assert "nli" in list_methods()
        params_class = get_params_class("nli")
        assert params_class.config_key == "nli_hypotheses"
        assert params_class.label_prefix == "n"
        assert params_class.requires_runner is False
        assert params_class.requires_embedder is False


class TestNliParams:
    """Test NliParams defaults and validation."""

    def test_default_values(self):
        """Should have sensible defaults."""
        params = NliParams()
        assert params.nli_model == "facebook/bart-large-mnli"
        assert params.aggregation == "mean"

    def test_params_fields(self):
        """Should have all required class vars."""
        assert NliParams.name == "nli"
        assert NliParams.config_key == "nli_hypotheses"
        assert NliParams.label_prefix == "n"

    def test_alternative_aggregations(self):
        """Should accept different aggregation methods."""
        params_max = NliParams(aggregation="max")
        params_noisy = NliParams(aggregation="noisy_or")
        assert params_max.aggregation == "max"
        assert params_noisy.aggregation == "noisy_or"


class TestNliScoring:
    """Test NLI scoring logic (uses mock pipeline)."""

    def test_score_text_nli_mean(self):
        """Should compute mean aggregation correctly."""

        # Mock pipeline that returns fixed scores
        class MockPipeline:
            def __call__(self, text, candidate_labels, multi_label=False):
                # Return scores in descending order (sorted by score)
                return {
                    "labels": ["hyp_b", "hyp_a"],
                    "scores": [0.8, 0.4],
                }

        result = score_text_nli(
            "Test text",
            ["hyp_a", "hyp_b"],
            MockPipeline(),
            aggregation="mean",
        )
        # Mean of 0.4 and 0.8 = 0.6
        assert abs(result["score"] - 0.6) < 0.001
        assert result["per_hypothesis"]["hyp_a"] == 0.4
        assert result["per_hypothesis"]["hyp_b"] == 0.8

    def test_score_text_nli_max(self):
        """Should compute max aggregation correctly."""

        class MockPipeline:
            def __call__(self, text, candidate_labels, multi_label=False):
                return {
                    "labels": ["hyp_b", "hyp_a"],
                    "scores": [0.9, 0.3],
                }

        result = score_text_nli(
            "Test text",
            ["hyp_a", "hyp_b"],
            MockPipeline(),
            aggregation="max",
        )
        assert result["score"] == 0.9

    def test_score_text_nli_noisy_or(self):
        """Should compute noisy-OR aggregation correctly."""

        class MockPipeline:
            def __call__(self, text, candidate_labels, multi_label=False):
                return {
                    "labels": ["hyp_a", "hyp_b"],
                    "scores": [0.5, 0.5],
                }

        result = score_text_nli(
            "Test text",
            ["hyp_a", "hyp_b"],
            MockPipeline(),
            aggregation="noisy_or",
        )
        # 1 - (1-0.5)(1-0.5) = 1 - 0.25 = 0.75
        assert abs(result["score"] - 0.75) < 0.001

    def test_scores_ordered_matches_input_order(self):
        """Scores should be in original hypothesis order."""

        class MockPipeline:
            def __call__(self, text, candidate_labels, multi_label=False):
                # Return in different order than input
                return {
                    "labels": ["third", "first", "second"],
                    "scores": [0.9, 0.1, 0.5],
                }

        result = score_text_nli(
            "Test",
            ["first", "second", "third"],
            MockPipeline(),
            aggregation="mean",
        )
        # scores_ordered should follow input order: first, second, third
        assert result["scores_ordered"] == [0.1, 0.5, 0.9]

    def test_invalid_aggregation_raises(self):
        """Should raise error for unknown aggregation."""

        class MockPipeline:
            def __call__(self, text, candidate_labels, multi_label=False):
                return {"labels": ["h"], "scores": [0.5]}

        with pytest.raises(ValueError, match="Unknown aggregation"):
            score_text_nli("Text", ["h"], MockPipeline(), aggregation="invalid")
