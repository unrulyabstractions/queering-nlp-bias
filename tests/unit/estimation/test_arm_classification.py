"""Tests for arm classification utilities."""

from __future__ import annotations

import pytest

from src.estimation.arm_types import (
    ArmKind,
    classify_arm,
    is_twig,
    get_parent_branch,
    get_twig_index,
    get_branch_index,
    is_baseline_arm,
    is_reference_arm,
    get_arm_sort_key,
    get_arm_name_from_index,
    sort_arm_names,
    get_arm_ancestry,
    get_downstream_arms,
)


class TestClassifyArm:
    """Tests for classify_arm function."""

    def test_classify_root(self):
        assert classify_arm("root") == ArmKind.ROOT

    def test_classify_trunk(self):
        assert classify_arm("trunk") == ArmKind.TRUNK

    def test_classify_branch(self):
        assert classify_arm("branch_1") == ArmKind.BRANCH
        assert classify_arm("branch_2") == ArmKind.BRANCH
        assert classify_arm("branch_10") == ArmKind.BRANCH

    def test_classify_twig(self):
        assert classify_arm("twig_1_b1") == ArmKind.TWIG
        assert classify_arm("twig_2_b1") == ArmKind.TWIG
        assert classify_arm("twig_1_b2") == ArmKind.TWIG

    def test_classify_unknown_defaults_to_branch(self):
        # Unknown names default to branch
        assert classify_arm("something_else") == ArmKind.BRANCH


class TestIsTwig:
    """Tests for is_twig function."""

    def test_twig_returns_true(self):
        assert is_twig("twig_1_b1") is True
        assert is_twig("twig_3_b2") is True

    def test_non_twig_returns_false(self):
        assert is_twig("root") is False
        assert is_twig("trunk") is False
        assert is_twig("branch_1") is False


class TestGetParentBranch:
    """Tests for get_parent_branch function."""

    def test_twig_returns_parent(self):
        assert get_parent_branch("twig_1_b1") == "branch_1"
        assert get_parent_branch("twig_2_b3") == "branch_3"

    def test_non_twig_returns_none(self):
        assert get_parent_branch("root") is None
        assert get_parent_branch("trunk") is None
        assert get_parent_branch("branch_1") is None

    def test_invalid_twig_format_returns_none(self):
        assert get_parent_branch("twig_invalid") is None


class TestGetTwigIndex:
    """Tests for get_twig_index function."""

    def test_twig_returns_index(self):
        assert get_twig_index("twig_1_b1") == 1
        assert get_twig_index("twig_5_b2") == 5

    def test_non_twig_returns_none(self):
        assert get_twig_index("branch_1") is None


class TestGetBranchIndex:
    """Tests for get_branch_index function."""

    def test_branch_returns_index(self):
        assert get_branch_index("branch_1") == 1
        assert get_branch_index("branch_5") == 5

    def test_twig_returns_branch_index(self):
        assert get_branch_index("twig_1_b1") == 1
        assert get_branch_index("twig_2_b3") == 3

    def test_non_branch_returns_none(self):
        assert get_branch_index("root") is None
        assert get_branch_index("trunk") is None


class TestIsBaselineArm:
    """Tests for is_baseline_arm function."""

    def test_root_is_baseline(self):
        assert is_baseline_arm("root") is True

    def test_trunk_is_baseline(self):
        assert is_baseline_arm("trunk") is True

    def test_branch_not_baseline(self):
        assert is_baseline_arm("branch_1") is False

    def test_twig_not_baseline(self):
        assert is_baseline_arm("twig_1_b1") is False


class TestIsReferenceArm:
    """Tests for is_reference_arm function."""

    def test_trunk_is_reference(self):
        assert is_reference_arm("trunk") is True

    def test_root_not_reference(self):
        assert is_reference_arm("root") is False

    def test_branch_not_reference(self):
        assert is_reference_arm("branch_1") is False


class TestGetArmSortKey:
    """Tests for get_arm_sort_key function."""

    def test_root_sorts_first(self):
        key = get_arm_sort_key("root")
        assert key[0] == 0  # Category 0

    def test_trunk_sorts_second(self):
        key = get_arm_sort_key("trunk")
        assert key[0] == 1  # Category 1

    def test_branches_sort_by_index(self):
        key1 = get_arm_sort_key("branch_1")
        key2 = get_arm_sort_key("branch_2")
        assert key1 < key2

    def test_twigs_sort_after_parent_branch(self):
        branch_key = get_arm_sort_key("branch_1")
        twig_key = get_arm_sort_key("twig_1_b1")
        # Both in category 2, same branch index
        assert branch_key < twig_key  # Branch has twig_idx=0


class TestGetArmNameFromIndex:
    """Tests for get_arm_name_from_index function."""

    def test_index_0_is_root(self):
        assert get_arm_name_from_index(0) == "root"

    def test_index_1_is_trunk(self):
        assert get_arm_name_from_index(1) == "trunk"

    def test_index_2_plus_are_branches(self):
        assert get_arm_name_from_index(2) == "branch_1"
        assert get_arm_name_from_index(3) == "branch_2"
        assert get_arm_name_from_index(5) == "branch_4"


class TestSortArmNames:
    """Tests for sort_arm_names function."""

    def test_sorts_correctly(self):
        names = ["branch_2", "root", "twig_1_b1", "trunk", "branch_1"]
        result = sort_arm_names(names)
        assert result == ["root", "trunk", "branch_1", "twig_1_b1", "branch_2"]

    def test_empty_list(self):
        assert sort_arm_names([]) == []


class TestGetArmAncestry:
    """Tests for get_arm_ancestry function."""

    def test_root_ancestry(self):
        assert get_arm_ancestry("root") == ["root"]

    def test_trunk_ancestry(self):
        assert get_arm_ancestry("trunk") == ["root", "trunk"]

    def test_branch_ancestry(self):
        assert get_arm_ancestry("branch_2") == ["root", "trunk", "branch_2"]

    def test_twig_ancestry(self):
        assert get_arm_ancestry("twig_1_b2") == ["root", "trunk", "branch_2", "twig_1_b2"]


class TestGetDownstreamArms:
    """Tests for get_downstream_arms function."""

    def test_root_has_all_downstream(self):
        all_arms = ["root", "trunk", "branch_1", "twig_1_b1"]
        downstream = get_downstream_arms("root", all_arms)
        assert "trunk" in downstream
        assert "branch_1" in downstream
        assert "twig_1_b1" in downstream

    def test_trunk_has_branches_downstream(self):
        all_arms = ["root", "trunk", "branch_1", "branch_2", "twig_1_b1"]
        downstream = get_downstream_arms("trunk", all_arms)
        assert "branch_1" in downstream
        assert "branch_2" in downstream
        assert "twig_1_b1" in downstream
        assert "root" not in downstream

    def test_branch_has_own_twigs_downstream(self):
        all_arms = ["trunk", "branch_1", "branch_2", "twig_1_b1", "twig_1_b2"]
        downstream = get_downstream_arms("branch_1", all_arms)
        assert "twig_1_b1" in downstream
        assert "twig_1_b2" not in downstream  # Different branch

    def test_twig_has_nothing_downstream(self):
        all_arms = ["trunk", "branch_1", "twig_1_b1", "twig_2_b1"]
        downstream = get_downstream_arms("twig_1_b1", all_arms)
        assert downstream == []
