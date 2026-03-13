"""Tests for BranchingNode class."""

from __future__ import annotations

import pytest

from src.common.branching_node import BranchingNode


class TestBranchingNodeBasics:
    """Tests for BranchingNode basic functionality."""

    def test_create_branching_node(self):
        node = BranchingNode(
            next_token_ids=(100, 200, 300),
            next_token_logprobs=(-1.0, -1.5, -2.0),
            branching_token_position=5,
        )
        assert node.next_token_ids == (100, 200, 300)
        assert node.next_token_logprobs == (-1.0, -1.5, -2.0)
        assert node.branching_token_position == 5

    def test_optional_fields_default_none(self):
        node = BranchingNode(
            next_token_ids=(100,),
            next_token_logprobs=(-1.0,),
            branching_token_position=3,
        )
        assert node.node_idx is None
        assert node.traj_idx is None
        assert node.vocab_logits is None
        assert node.forks_idx is None

    def test_with_traj_idx(self):
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            traj_idx=[0, 1, 2],
        )
        assert node.traj_idx == [0, 1, 2]

    def test_with_vocab_logits(self):
        vocab_logits = [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.3, 0.4, 0.5],
        ]
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            vocab_logits=vocab_logits,
        )
        assert node.vocab_logits == vocab_logits

    def test_with_forks_idx(self):
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            forks_idx=[0, 1],
        )
        assert node.forks_idx == [0, 1]


class TestBranchingNodeSerialization:
    """Tests for BranchingNode serialization."""

    def test_to_dict_basic(self):
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        d = node.to_dict()
        assert d["next_token_ids"] == [100, 200]
        assert d["next_token_logprobs"] == [-1.0, -1.5]
        assert d["branching_token_position"] == 5

    def test_to_dict_summarizes_vocab_logits(self):
        # vocab_logits can be large - should be summarized
        vocab_logits = [
            [0.1] * 1000,  # Simulating vocab size of 1000
            [0.2] * 1000,
        ]
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            vocab_logits=vocab_logits,
        )
        d = node.to_dict()
        # Should be summarized as count
        assert d["vocab_logits"] == "[2000 items]"

    def test_to_dict_hook_applied(self):
        vocab_logits = [[0.1, 0.2], [0.3, 0.4]]
        node = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
            vocab_logits=vocab_logits,
        )
        d = node.to_dict()
        # Total items = 2 * 2 = 4
        assert d["vocab_logits"] == "[4 items]"

    def test_from_dict_basic(self):
        data = {
            "next_token_ids": [100, 200],
            "next_token_logprobs": [-1.0, -1.5],
            "branching_token_position": 5,
        }
        node = BranchingNode.from_dict(data)
        assert node.next_token_ids == (100, 200)
        assert node.next_token_logprobs == (-1.0, -1.5)
        assert node.branching_token_position == 5

    def test_from_dict_with_optional_fields(self):
        data = {
            "next_token_ids": [100, 200],
            "next_token_logprobs": [-1.0, -1.5],
            "branching_token_position": 5,
            "node_idx": 0,
            "traj_idx": [0, 1],
            "forks_idx": [0],
        }
        node = BranchingNode.from_dict(data)
        assert node.node_idx == 0
        assert node.traj_idx == [0, 1]
        assert node.forks_idx == [0]


class TestBranchingNodeId:
    """Tests for BranchingNode ID generation."""

    def test_same_values_same_id(self):
        node1 = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        node2 = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        assert node1.get_id() == node2.get_id()

    def test_different_values_different_id(self):
        node1 = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        node2 = BranchingNode(
            next_token_ids=(100, 300),  # Different token
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        assert node1.get_id() != node2.get_id()

    def test_different_position_different_id(self):
        node1 = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=5,
        )
        node2 = BranchingNode(
            next_token_ids=(100, 200),
            next_token_logprobs=(-1.0, -1.5),
            branching_token_position=6,  # Different position
        )
        assert node1.get_id() != node2.get_id()
