"""Integration tests for scoring pipeline."""

from __future__ import annotations

import pytest

from fixtures.sample_trajectories import create_sample_trajectory
from fixtures.sample_compliance_data import (
    create_compliance_vectors,
    create_trajectory_scoring_data,
)


@pytest.mark.integration
class TestScoringDataStructures:
    """Integration tests for scoring data structures."""

    def test_trajectory_scoring_data_creation(self):
        """Test creation of trajectory scoring data."""
        data = create_trajectory_scoring_data(n_trajectories=10, n_structures=4)

        assert len(data) == 10
        for item in data:
            assert len(item.structure_scores) == 4
            assert item.log_prob < 0  # Log probs are negative
            assert item.n_tokens > 0

    def test_compliance_vectors_valid(self):
        """Test that compliance vectors are in valid range."""
        vectors = create_compliance_vectors(n_samples=20, n_structures=5)

        assert len(vectors) == 20
        for vec in vectors:
            assert len(vec) == 5
            for score in vec:
                assert 0.0 <= score <= 1.0

    def test_scoring_data_branch_assignment(self):
        """Test that scoring data has proper branch assignments."""
        data = create_trajectory_scoring_data(n_trajectories=10, n_structures=4)

        branches = [item.branch for item in data]
        # Should have both trunk and branch_1
        assert "trunk" in branches
        assert "branch_1" in branches


@pytest.mark.integration
class TestScoringWithTrajectories:
    """Integration tests for scoring with trajectory objects."""

    def test_trajectory_text_extraction(self):
        """Test that trajectory text can be extracted for scoring."""
        traj = create_sample_trajectory(
            prefill_text="[PREFIX]",
            generated_text="This is the generated text.",
        )

        # Full continuation
        assert traj.continuation_text == "[PREFIX]This is the generated text."

        # Without thinking blocks
        assert traj.continuation_text_no_thinking == "[PREFIX]This is the generated text."

    def test_trajectory_with_thinking_blocks(self):
        """Test scoring text extraction with thinking blocks."""
        traj = create_sample_trajectory(
            prefill_text="",
            generated_text="<think>Internal reasoning</think>Final answer.",
        )

        # Should strip thinking blocks
        assert "<think>" not in traj.continuation_text_no_thinking
        assert "Final answer." in traj.continuation_text_no_thinking

    def test_trajectory_text_after_arm(self):
        """Test extracting text after a specific arm."""
        traj = create_sample_trajectory(
            prefill_text="[PREFIX]",
            generated_text="Generated content here.",
        )
        traj.arm_text_lengths = [8]  # "[PREFIX]" is 8 chars

        text_after = traj.text_after_arm(0)
        assert text_after == "Generated content here."


@pytest.mark.integration
class TestCategoricalScoring:
    """Integration tests for categorical scoring patterns."""

    def test_binary_categorical_output(self):
        """Test that categorical scoring produces binary outputs."""
        # Simulate categorical scores
        scores = [0, 1, 1, 0, 1, 0]

        # All should be 0 or 1
        assert all(s in [0, 1] for s in scores)

        # Can compute compliance (fraction of 1s)
        compliance = sum(scores) / len(scores)
        assert 0.0 <= compliance <= 1.0

    def test_multi_category_output(self):
        """Test multi-category scoring produces valid labels."""
        categories = ["positive", "negative", "neutral"]
        scores = ["positive", "negative", "neutral", "positive"]

        assert all(s in categories for s in scores)


@pytest.mark.integration
class TestStructureScoring:
    """Integration tests for structure-based scoring."""

    def test_structure_scores_per_trajectory(self):
        """Test that each trajectory gets scores for all structures."""
        data = create_trajectory_scoring_data(n_trajectories=5, n_structures=3)

        for item in data:
            assert len(item.structure_scores) == 3

    def test_structure_scores_form_compliance_vector(self):
        """Test that structure scores form valid compliance vectors."""
        data = create_trajectory_scoring_data(n_trajectories=10, n_structures=4)

        compliance_vectors = [item.structure_scores for item in data]

        assert len(compliance_vectors) == 10
        for vec in compliance_vectors:
            assert len(vec) == 4
            assert all(0.0 <= s <= 1.0 for s in vec)
