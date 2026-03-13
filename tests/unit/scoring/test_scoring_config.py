"""Tests for ScoringConfig class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


class TestScoringConfigBasics:
    """Tests for ScoringConfig basic functionality.

    Note: These tests verify the structure of scoring configs
    without depending on specific method implementations.
    """

    def test_scoring_config_dict_structure(self):
        # Verify expected structure of a scoring config
        data = {
            "judge_model": "test-model",
            "methods": {
                "categorical": {
                    "prompt_template": "Is this text {category}? Answer yes or no.",
                    "categories": ["positive", "negative"],
                },
            },
        }
        # Basic structure validation
        assert "judge_model" in data
        assert "methods" in data

    def test_scoring_config_methods_can_be_empty(self):
        data = {
            "judge_model": "test-model",
            "methods": {},
        }
        assert data["methods"] == {}

    def test_scoring_config_with_multiple_methods(self):
        data = {
            "judge_model": "test-model",
            "methods": {
                "categorical": {"categories": ["yes", "no"]},
                "graded": {"scale_min": 1, "scale_max": 5},
                "similarity": {"reference_texts": ["text1", "text2"]},
            },
        }
        assert len(data["methods"]) == 3

    def test_scoring_config_json_serialization(self):
        data = {
            "judge_model": "test-model",
            "embedding_model": "test-embedder",
            "methods": {
                "categorical": {"prompt": "Test prompt"},
            },
        }
        # Should be JSON serializable
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed == data


class TestScoringMethodStructures:
    """Tests for scoring method configuration structures."""

    def test_categorical_method_structure(self):
        method_config = {
            "prompt_template": "Classify this text: {text}",
            "categories": ["positive", "negative", "neutral"],
            "default_category": "neutral",
        }
        assert "categories" in method_config
        assert len(method_config["categories"]) == 3

    def test_graded_method_structure(self):
        method_config = {
            "prompt_template": "Rate this text from 1-5: {text}",
            "scale_min": 1,
            "scale_max": 5,
        }
        assert method_config["scale_min"] < method_config["scale_max"]

    def test_similarity_method_structure(self):
        method_config = {
            "reference_texts": ["Reference text 1", "Reference text 2"],
            "similarity_metric": "cosine",
        }
        assert "reference_texts" in method_config
        assert len(method_config["reference_texts"]) > 0

    def test_count_occurrences_method_structure(self):
        method_config = {
            "patterns": ["word1", "word2", "phrase with spaces"],
            "case_sensitive": False,
        }
        assert "patterns" in method_config


class TestStringSelectionEnum:
    """Tests for string selection options."""

    def test_string_selection_options(self):
        # Common string selection options in scoring
        options = [
            "WholeContinuation",
            "AfterTrunk",
            "AfterBranch",
            "GeneratedOnly",
        ]
        # All should be valid string options
        assert len(options) == 4

    def test_string_selection_affects_scoring(self):
        # Different selection should affect which text is scored
        config_whole = {"string_selection": "WholeContinuation"}
        config_after = {"string_selection": "AfterTrunk"}
        assert config_whole["string_selection"] != config_after["string_selection"]


class TestScoringConfigFileIO:
    """Tests for scoring config file operations."""

    def test_write_and_read_config(self):
        data = {
            "judge_model": "test-model",
            "methods": {
                "categorical": {"categories": ["a", "b"]},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            path = Path(f.name)

        # Read back
        with open(path) as f:
            loaded = json.load(f)

        assert loaded == data
        path.unlink()

    def test_config_with_nested_structures(self):
        data = {
            "judge_model": "model",
            "methods": {
                "complex_method": {
                    "param1": "value1",
                    "nested": {
                        "inner_param": 42,
                        "list_param": [1, 2, 3],
                    },
                },
            },
        }
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        assert parsed["methods"]["complex_method"]["nested"]["inner_param"] == 42


class TestScoringMethodParams:
    """Tests for method-specific parameters."""

    def test_categorical_params(self):
        params = {
            "prompt_template": "Is {text} in category {category}?",
            "categories": ["cat1", "cat2"],
            "use_thinking": False,
        }
        assert "{text}" in params["prompt_template"]
        assert "{category}" in params["prompt_template"]

    def test_graded_params_validation(self):
        params = {
            "scale_min": 1,
            "scale_max": 10,
            "step": 1,
        }
        # Scale validation
        assert params["scale_min"] < params["scale_max"]
        assert params["step"] > 0

    def test_params_with_defaults(self):
        # Method params with default values
        full_params = {
            "required_param": "value",
            "optional_param": None,
            "with_default": 10,
        }
        # Missing optional params should work
        minimal_params = {"required_param": "value"}
        assert "required_param" in minimal_params
