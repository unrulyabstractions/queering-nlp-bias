"""Tests for GenerationConfig class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.generation.generation_config import GenerationConfig, MethodParamsOverride
from src.common.experiment_types import GenerationArm


class TestGenerationConfigBasics:
    """Tests for GenerationConfig basic functionality."""

    def test_create_minimal_config(self):
        config = GenerationConfig(prompt="Test prompt")
        assert config.prompt == "Test prompt"
        assert config.trunk == ""
        assert config.branches == []

    def test_create_full_config(self):
        config = GenerationConfig(
            prompt="Test prompt",
            model="test-model",
            trunk="Trunk text",
            branches=["Branch 1", "Branch 2"],
            temperature=0.8,
            max_new_tokens=100,
        )
        assert config.model == "test-model"
        assert config.trunk == "Trunk text"
        assert len(config.branches) == 2
        assert config.temperature == 0.8


class TestGenerationConfigGetArms:
    """Tests for get_arms method."""

    def test_get_arms_trunk_only(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="Trunk text",
            branches=[],
        )
        arms = config.get_arms()
        assert len(arms) == 2  # root + trunk
        assert arms[0].name == "root"
        assert arms[1].name == "trunk"

    def test_get_arms_with_branches(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="Trunk",
            branches=["Branch A", "Branch B"],
        )
        arms = config.get_arms()
        assert len(arms) == 4  # root + trunk + 2 branches
        assert arms[0].name == "root"
        assert arms[1].name == "trunk"
        assert arms[2].name == "branch_1"
        assert arms[3].name == "branch_2"

    def test_get_arms_prefill_construction(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="[Trunk]",
            branches=["[Branch1]", "[Branch2]"],
        )
        arms = config.get_arms()
        # Root has empty prefill
        assert arms[0].prefill == ""
        # Trunk has trunk text
        assert arms[1].prefill == "[Trunk]"
        # Branches have trunk + branch
        assert arms[2].prefill == "[Trunk][Branch1]"
        assert arms[3].prefill == "[Trunk][Branch2]"

    def test_get_arms_with_skip_prefix(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="Trunk",
            branches=["Branch"],
        )
        arms = config.get_arms(skip_prefix="<skip>")
        # All arms should start with skip_prefix
        assert arms[0].prefill == "<skip>"
        assert arms[1].prefill == "<skip>Trunk"
        assert arms[2].prefill == "<skip>TrunkBranch"

    def test_get_arms_parent_indices(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="Trunk",
            branches=["Branch"],
        )
        arms = config.get_arms()
        # Root has no parent (implicit None or 0)
        assert arms[1].parent_idx == 0  # trunk's parent is root
        assert arms[2].parent_idx == 1  # branch's parent is trunk

    def test_get_arms_with_twigs(self):
        config = GenerationConfig(
            prompt="Test",
            trunk="Trunk",
            branches=["Branch1", "Branch2"],
            twig_variations=["TwigA", "TwigB"],
        )
        arms = config.get_arms()
        # root + trunk + 2*(branch + 2 twigs) = 2 + 2*3 = 8
        assert len(arms) == 8
        # Check twig naming
        twig_names = [a.name for a in arms if "twig" in a.name]
        assert "twig_1_b1" in twig_names
        assert "twig_2_b1" in twig_names
        assert "twig_1_b2" in twig_names
        assert "twig_2_b2" in twig_names


class TestMethodParamsOverride:
    """Tests for MethodParamsOverride class."""

    def test_create_override(self):
        override = MethodParamsOverride(overrides={"samples_per_arm": 20})
        assert override.overrides == {"samples_per_arm": 20}

    def test_apply_to_params(self):
        override = MethodParamsOverride(overrides={"samples_per_arm": 20})

        # Create a mock params object
        class MockParams:
            samples_per_arm: int = 10

        params = MockParams()
        override.apply_to(params)
        assert params.samples_per_arm == 20

    def test_apply_ignores_nonexistent_attrs(self):
        override = MethodParamsOverride(overrides={"nonexistent_field": 100})

        class MockParams:
            existing_field: int = 10

        params = MockParams()
        # Should not raise, just ignore
        override.apply_to(params)
        assert params.existing_field == 10


class TestGenerationConfigSerialization:
    """Tests for GenerationConfig serialization."""

    def test_to_dict(self):
        config = GenerationConfig(
            prompt="Test prompt",
            trunk="Trunk",
            branches=["B1"],
        )
        d = config.to_dict()
        assert d["prompt"] == "Test prompt"
        assert d["trunk"] == "Trunk"
        assert d["branches"] == ["B1"]

    def test_from_dict(self):
        data = {
            "prompt": "Test prompt",
            "trunk": "Trunk text",
            "branches": ["Branch 1", "Branch 2"],
            "temperature": 0.9,
        }
        config = GenerationConfig.from_dict(data)
        assert config.prompt == "Test prompt"
        assert config.trunk == "Trunk text"
        assert len(config.branches) == 2
        assert config.temperature == 0.9

    def test_from_dict_with_method_params(self):
        data = {
            "prompt": "Test",
            "method_params": {
                "simple-sampling": {"overrides": {"samples_per_arm": 30}},
            },
        }
        config = GenerationConfig.from_dict(data)
        assert "simple-sampling" in config.method_params
        assert config.method_params["simple-sampling"].overrides["samples_per_arm"] == 30

    def test_load_from_file(self):
        data = {
            "prompt": "Test prompt from file",
            "trunk": "File trunk",
            "branches": ["File branch"],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            config = GenerationConfig.load(f.name)

        assert config.prompt == "Test prompt from file"
        assert config.trunk == "File trunk"
        Path(f.name).unlink()

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            GenerationConfig.load("/nonexistent/path.json")


class TestGenerationConfigGetParams:
    """Tests for get_params method."""

    def test_get_params_returns_params_object(self):
        config = GenerationConfig(prompt="Test")
        # This test depends on registered methods
        # If no method is registered, this will fail
        # For now, test that it calls through properly
        try:
            params = config.get_params("simple-sampling")
            assert params is not None
        except KeyError:
            # Method not registered - skip test
            pytest.skip("simple-sampling method not registered")

    def test_get_params_applies_overrides(self):
        config = GenerationConfig(
            prompt="Test",
            method_params={
                "simple-sampling": MethodParamsOverride(
                    overrides={"samples_per_arm": 50}
                ),
            },
        )
        try:
            params = config.get_params("simple-sampling")
            assert params.samples_per_arm == 50
        except KeyError:
            pytest.skip("simple-sampling method not registered")
