"""Tests for TokenTree class."""

from __future__ import annotations

import pytest

from src.common.token_tree import TokenTree
from src.common.token_trajectory import TokenTrajectory
from src.common.branching_node import BranchingNode
from src.common.binary_fork import BinaryFork
from fixtures.sample_trajectories import (
    create_divergent_trajectories,
    create_trajectory_pair_for_fork,
    create_trajectory_batch,
    create_sample_trajectory,
)


class TestTokenTreeFromTrajectories:
    """Tests for TokenTree.from_trajectories class method."""

    def test_from_trajectories_empty(self):
        tree = TokenTree.from_trajectories([])
        assert tree.trajs == ()
        assert tree.nodes == ()
        assert tree.forks == ()

    def test_from_trajectories_single(self):
        traj = create_sample_trajectory(n_tokens=10)
        tree = TokenTree.from_trajectories([traj])
        assert len(tree.trajs) == 1
        # No branching with single trajectory
        assert len(tree.nodes) == 0

    def test_from_trajectories_divergent_creates_nodes(self):
        trajs = create_divergent_trajectories(
            n_trajectories=3,
            diverge_at=5,
            total_length=10,
        )
        # Provide group assignments
        groups_per_traj = [(0,), (1,), (2,)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
        )
        assert len(tree.trajs) == 3
        # Should have a branching node at divergence point
        assert len(tree.nodes) >= 1

    def test_from_trajectories_with_groups(self):
        trajs = create_divergent_trajectories(
            n_trajectories=4,
            diverge_at=5,
        )
        # Group first two together, last two together
        groups_per_traj = [(0,), (0,), (1,), (1,)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
        )
        assert tree.groups == (0, 1)
        assert tree.n_groups == 2

    def test_from_trajectories_with_fork_arms(self):
        trajs = create_divergent_trajectories(
            n_trajectories=4,
            diverge_at=5,
        )
        groups_per_traj = [(0,), (0,), (1,), (1,)]
        fork_arms = [(0, 1)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
            fork_arms=fork_arms,
        )
        assert tree.fork_arms == ((0, 1),)

    def test_from_trajectories_sets_arm_idx(self):
        trajs = create_divergent_trajectories(n_trajectories=2)
        groups_per_traj = [(0,), (1,)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
        )
        assert tree.trajs[0].arm_idx == (0,)
        assert tree.trajs[1].arm_idx == (1,)


class TestTokenTreeProperties:
    """Tests for TokenTree properties."""

    def test_groups_empty_tree(self):
        tree = TokenTree(trajs=())
        assert tree.groups == ()

    def test_groups_with_trajectories(self):
        traj1 = create_sample_trajectory(arm_idx=(0,))
        traj2 = create_sample_trajectory(arm_idx=(1,))
        traj3 = create_sample_trajectory(arm_idx=(0, 1))
        tree = TokenTree(trajs=(traj1, traj2, traj3))
        assert tree.groups == (0, 1)

    def test_n_groups(self):
        traj1 = create_sample_trajectory(arm_idx=(0,))
        traj2 = create_sample_trajectory(arm_idx=(1,))
        tree = TokenTree(trajs=(traj1, traj2))
        assert tree.n_groups == 2


class TestTokenTreeAddTrajectory:
    """Tests for add_trajectory method."""

    def test_add_trajectory_returns_new_tree(self):
        traj1 = create_sample_trajectory(n_tokens=10)
        tree = TokenTree.from_trajectories([traj1], groups_per_traj=[(0,)])

        traj2 = create_sample_trajectory(n_tokens=10)
        new_tree = tree.add_trajectory(traj2, arm_idx=[1])

        # Original tree unchanged
        assert len(tree.trajs) == 1
        # New tree has both
        assert len(new_tree.trajs) == 2

    def test_add_trajectory_new_group(self):
        traj1 = create_sample_trajectory(arm_idx=(0,))
        tree = TokenTree(trajs=(traj1,))

        traj2 = create_sample_trajectory()
        new_tree = tree.add_trajectory(traj2, arm_idx=[1])

        assert new_tree.n_groups == 2


class TestTokenTreeAddForkBetweenGroups:
    """Tests for add_fork_between_groups method."""

    def test_add_fork_between_groups_basic(self):
        trajs = create_divergent_trajectories(
            n_trajectories=2,
            diverge_at=5,
        )
        groups_per_traj = [(0,), (1,)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
            fork_arms=[],  # No forks initially
        )
        assert tree.fork_arms == ()

        new_tree = tree.add_fork_between_groups((0, 1))
        assert (0, 1) in new_tree.fork_arms


class TestTokenTreePopHeavy:
    """Tests for pop_heavy method."""

    def test_pop_heavy_clears_trajectory_logits(self):
        from tests.fixtures.sample_trajectories import create_trajectory_with_full_logits

        traj = create_trajectory_with_full_logits()
        tree = TokenTree(trajs=(traj,))

        assert tree.trajs[0].full_logits is not None
        tree.pop_heavy()
        assert tree.trajs[0].full_logits is None

    def test_pop_heavy_clears_node_vocab_logits(self):
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            vocab_logits=[[0.1, 0.2, 0.3]],
        )
        tree = TokenTree(trajs=(), nodes=(node,))

        tree.pop_heavy()
        assert tree.nodes[0].vocab_logits is None


class TestTokenTreeFromDict:
    """Tests for TokenTree.from_dict class method."""

    def test_from_dict_basic(self):
        data = {
            "trajs": [
                {"token_ids": [1, 2, 3], "logprobs": [0.0, -1.0, -2.0], "logits": [0.0, 1.0, 2.0]},
            ],
            "nodes": None,
            "forks": None,
            "trunk_length": 5,
            "prompt_length": 3,
        }
        tree = TokenTree.from_dict(data)
        assert len(tree.trajs) == 1
        assert tree.trunk_length == 5
        assert tree.prompt_length == 3

    def test_from_dict_with_nodes(self):
        data = {
            "trajs": [
                {"token_ids": [1, 2, 3], "logprobs": [0.0, -1.0, -2.0], "logits": [0.0, 1.0, 2.0]},
            ],
            "nodes": [
                {
                    "next_token_ids": [100, 200],
                    "next_token_logprobs": [-1.0, -1.5],
                    "branching_token_position": 5,
                },
            ],
            "forks": None,
        }
        tree = TokenTree.from_dict(data)
        assert len(tree.nodes) == 1
        assert isinstance(tree.nodes[0], BranchingNode)

    def test_from_dict_with_forks(self):
        data = {
            "trajs": [],
            "forks": [
                {
                    "next_token_ids": [100, 200],
                    "next_token_logprobs": [-1.0, -1.5],
                },
            ],
            "fork_arms": [[0, 1]],
        }
        tree = TokenTree.from_dict(data)
        assert len(tree.forks) == 1
        assert isinstance(tree.forks[0], BinaryFork)
        assert tree.fork_arms == ((0, 1),)

    def test_from_dict_preserves_trajectory_objects(self):
        traj = TokenTrajectory(
            token_ids=[1, 2, 3],
            logprobs=[0.0, -1.0, -2.0],
            logits=[0.0, 1.0, 2.0],
        )
        data = {"trajs": [traj]}
        tree = TokenTree.from_dict(data)
        assert tree.trajs[0] is traj


class TestTokenTreeDivergenceDetection:
    """Tests for divergence point detection in tree building."""

    def test_divergence_at_expected_position(self):
        # Create trajectories that diverge at position 5
        trajs = create_divergent_trajectories(
            n_trajectories=2,
            diverge_at=5,
            total_length=10,
        )
        groups_per_traj = [(0,), (1,)]
        tree = TokenTree.from_trajectories(
            trajs,
            groups_per_traj=groups_per_traj,
        )

        assert len(tree.nodes) >= 1
        # First node should be at position 5
        assert tree.nodes[0].branching_token_position == 5

    def test_multiple_divergence_points(self):
        # Create trajectories with multiple divergence points
        # First diverge at 3, then subgroups diverge at 6
        shared_prefix = [100, 101, 102]  # Positions 0-2

        trajs = []
        # Group A - diverges at position 6
        for i in range(2):
            tokens = shared_prefix + [200, 201, 202] + [300 + i * 10 + j for j in range(4)]
            logprobs = [0.0] + [-1.0] * (len(tokens) - 1)
            traj = TokenTrajectory(token_ids=tokens, logprobs=logprobs, logits=logprobs.copy())
            trajs.append(traj)

        # Group B - diverges at position 6 differently
        for i in range(2):
            tokens = shared_prefix + [400, 401, 402] + [500 + i * 10 + j for j in range(4)]
            logprobs = [0.0] + [-1.0] * (len(tokens) - 1)
            traj = TokenTrajectory(token_ids=tokens, logprobs=logprobs, logits=logprobs.copy())
            trajs.append(traj)

        groups_per_traj = [(0,), (0,), (1,), (1,)]
        tree = TokenTree.from_trajectories(trajs, groups_per_traj=groups_per_traj)

        # Should have divergence nodes
        assert len(tree.nodes) >= 1


class TestTokenTreeForkCreation:
    """Tests for fork creation between groups."""

    def test_fork_creation_between_specified_groups(self):
        traj_a, traj_b = create_trajectory_pair_for_fork(
            shared_length=5,
            total_length=10,
            arm_indices=((0,), (1,)),
        )

        tree = TokenTree.from_trajectories(
            [traj_a, traj_b],
            groups_per_traj=[(0,), (1,)],
            fork_arms=[(0, 1)],
        )

        # Should have a fork between the groups
        assert len(tree.forks) >= 1

    def test_no_forks_without_fork_arms(self):
        traj_a, traj_b = create_trajectory_pair_for_fork()

        tree = TokenTree.from_trajectories(
            [traj_a, traj_b],
            groups_per_traj=[(0,), (1,)],
            fork_arms=[],  # No fork arms specified
        )

        # Should have no forks
        assert len(tree.forks) == 0


class TestTokenTreeWithBatch:
    """Tests using trajectory batch factory."""

    def test_batch_creates_valid_tree(self):
        batch = create_trajectory_batch(
            n_groups=3,
            trajs_per_group=2,
            shared_length=5,
            total_length=10,
        )

        tree = TokenTree.from_trajectories(
            batch.trajectories,
            groups_per_traj=batch.groups_per_traj,
            fork_arms=batch.fork_arms,
        )

        assert len(tree.trajs) == 6
        assert tree.n_groups == 3
