"""Integration tests for generation pipeline."""

from __future__ import annotations

import pytest

from src.common.token_tree import TokenTree
from src.generation.generation_config import GenerationConfig
from fixtures.mock_model_runner import MockModelRunner, create_mock_runner
from fixtures.sample_trajectories import create_sample_trajectory


@pytest.mark.integration
class TestGenerationPipelineBasics:
    """Basic integration tests for the generation pipeline."""

    def test_mock_runner_generates_trajectories(self):
        """Test that mock runner can generate trajectory-like objects."""
        runner = create_mock_runner()
        traj = runner.generate_trajectory_from_prompt(
            prompt="Test prompt",
            max_new_tokens=10,
            temperature=1.0,
        )
        assert traj is not None
        assert len(traj.token_ids) > 0
        assert len(traj.logprobs) == len(traj.token_ids)

    def test_config_arm_building(self):
        """Test that config correctly builds arm structures."""
        config = GenerationConfig(
            prompt="Test prompt",
            trunk="[TRUNK]",
            branches=["[BRANCH1]", "[BRANCH2]"],
        )
        arms = config.get_arms()

        # Should have root, trunk, and 2 branches
        assert len(arms) == 4
        assert arms[0].name == "root"
        assert arms[1].name == "trunk"
        assert arms[2].name == "branch_1"
        assert arms[3].name == "branch_2"

    def test_arm_prefills_accumulate_correctly(self):
        """Test that arm prefills accumulate correctly."""
        config = GenerationConfig(
            prompt="Test",
            trunk="A",
            branches=["B", "C"],
        )
        arms = config.get_arms()

        assert arms[0].prefill == ""  # root
        assert arms[1].prefill == "A"  # trunk
        assert arms[2].prefill == "AB"  # branch_1
        assert arms[3].prefill == "AC"  # branch_2


@pytest.mark.integration
class TestTreeBuilding:
    """Integration tests for tree building from trajectories."""

    def test_tree_from_single_trajectory(self):
        """Test tree creation from a single trajectory."""
        traj = create_sample_trajectory(n_tokens=10)
        tree = TokenTree.from_trajectories([traj], groups_per_traj=[(0,)])

        assert len(tree.trajs) == 1
        assert tree.trajs[0].arm_idx == (0,)

    def test_tree_from_multiple_groups(self):
        """Test tree creation with multiple trajectory groups."""
        trajs = [
            create_sample_trajectory(n_tokens=10, arm_idx=(0,)),
            create_sample_trajectory(n_tokens=10, arm_idx=(0,)),
            create_sample_trajectory(n_tokens=10, arm_idx=(1,)),
            create_sample_trajectory(n_tokens=10, arm_idx=(1,)),
        ]
        groups = [(0,), (0,), (1,), (1,)]

        tree = TokenTree.from_trajectories(trajs, groups_per_traj=groups)

        assert len(tree.trajs) == 4
        assert tree.n_groups == 2

    def test_tree_tracks_trunk_length(self):
        """Test that tree tracks trunk length correctly."""
        trajs = [create_sample_trajectory(n_tokens=10)]
        trunk_tokens = [1, 2, 3, 4, 5]

        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=[(0,)],
            trunk=trunk_tokens,
        )

        assert tree.trunk_length == 5


@pytest.mark.integration
class TestGenerationWithMockRunner:
    """Integration tests using mock runner for generation."""

    def test_generate_with_prefill(self, integration_mock_runner):
        """Test generation with prefill text."""
        traj = integration_mock_runner.generate_trajectory_from_prompt(
            prompt="Test prompt",
            prefilling="[PREFILL]",
            max_new_tokens=10,
        )

        assert traj.prefill_text == "[PREFILL]"

    def test_multiple_generations(self, integration_mock_runner):
        """Test multiple sequential generations."""
        trajs = []
        for i in range(5):
            traj = integration_mock_runner.generate_trajectory_from_prompt(
                prompt=f"Prompt {i}",
                max_new_tokens=10,
            )
            trajs.append(traj)

        assert len(trajs) == 5
        # Each should be different (different pre-configured responses)
        for i, traj in enumerate(trajs):
            assert traj.generated_text == f"Generated response {i}"

    def test_tree_from_generated_trajectories(self, integration_mock_runner):
        """Test building a tree from generated trajectories."""
        trajs = []
        groups = []

        for group in range(2):
            for _ in range(3):
                traj = integration_mock_runner.generate_trajectory_from_prompt(
                    prompt="Test",
                    max_new_tokens=10,
                )
                trajs.append(traj)
                groups.append((group,))

        tree = TokenTree.from_trajectories(trajs, groups_per_traj=groups)

        assert len(tree.trajs) == 6
        assert tree.n_groups == 2


@pytest.mark.integration
class TestTrajectoryIndices:
    """Integration tests for trajectory indexing."""

    def test_arm_indices_correct_after_tree_build(self):
        """Test that arm indices are correctly set after tree building."""
        trajs = [
            create_sample_trajectory(n_tokens=10),
            create_sample_trajectory(n_tokens=10),
            create_sample_trajectory(n_tokens=10),
        ]
        groups = [(0,), (1,), (1,)]

        tree = TokenTree.from_trajectories(trajs, groups_per_traj=groups)

        assert tree.trajs[0].arm_idx == (0,)
        assert tree.trajs[1].arm_idx == (1,)
        assert tree.trajs[2].arm_idx == (1,)

    def test_traj_idx_set_after_tree_build(self):
        """Test that traj_idx is correctly set on trajectories."""
        trajs = [create_sample_trajectory() for _ in range(5)]
        groups = [(i,) for i in range(5)]

        tree = TokenTree.from_trajectories(trajs, groups_per_traj=groups)

        for i, traj in enumerate(tree.trajs):
            assert traj.traj_idx == i
