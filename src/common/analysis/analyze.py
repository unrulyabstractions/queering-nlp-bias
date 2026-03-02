"""Token tree analysis entry point and per-component analyzers.

Provides analyze_token_tree and helper functions for analyzing
trajectories, forks, and nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .builders import build_fork_analysis, build_node_analysis
from .metrics import TrajectoryAnalysis

if TYPE_CHECKING:
    from ..token_tree import TokenTree


def analyze_token_tree(tree: TokenTree) -> None:
    """Populate the ``analysis`` field on tree, trajectories, nodes, and forks.

    Mutates *tree* in place, setting tree.analysis to the StructureSystemAnalysis.

    Returns:
        StructureSystemAnalysis if tree has forks and groups, else None.
        Contains per-node cores/orientations based on trajectory probabilities.
    """
    _analyze_trajectories_basic(tree)
    _analyze_forks(tree)
    _analyze_nodes_basic(tree)


def _analyze_trajectories_basic(tree: TokenTree) -> None:
    """First pass: compute basic trajectory metrics.

    Computes metrics for full trajectory, and if trunk_length is set,
    also computes trunk_only and continuation_only metrics.
    """
    for i, traj in enumerate(tree.trajs):
        traj.analysis = TrajectoryAnalysis.from_trajectory(
            traj_idx=i, traj=traj, trunk_length=tree.trunk_length
        )


def _analyze_forks(tree: TokenTree) -> None:
    """Analyze all forks in the tree."""
    if not tree.forks:
        return
    for i, fork in enumerate(tree.forks):
        fork.analysis = build_fork_analysis(i, fork)


def _analyze_nodes_basic(tree: TokenTree) -> None:
    """First pass: compute basic node metrics."""
    if not tree.nodes:
        return
    for i, node in enumerate(tree.nodes):
        node.analysis = build_node_analysis(i, node, tree)
