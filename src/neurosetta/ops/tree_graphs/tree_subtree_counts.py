"""Per-vertex subtree counts on a NeuRosetta tree."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    subgraph_branch_count,
    subgraph_edge_count,
    subgraph_leaf_count,
    subgraph_vertex_count,
)
from .._doc_helpers import enrich_tree_graph_docstrings


def get_subtree_leaf_count(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Count descendant leaves in the subtree of every node.

    A leaf is a node with no children. The node itself counts, so a leaf's
    subtree has leaf count 1.

    Parameters
    ----------
    tree : _Tree
        Neuron tree. Must be a directed rooted tree.
    bind : bool, optional
        If True, store vertex property ``subtree_leaf_count`` and return None.
        If False, return an int64 array. By default False.

    Returns
    -------
    ndarray or None
        One count per node when ``bind`` is False.
    """
    return subgraph_leaf_count(tree.graph, bind=bind)


def get_subtree_edge_count(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Count edges inside the subtree of every node.

    The edge that connects a node to its parent is outside the subtree.
    A leaf has edge count 0. On a valid tree,
    ``vertex_count = edge_count + 1`` at every node.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_edge_count`` and return None. By default False.

    Returns
    -------
    ndarray or None
        One count per node when ``bind`` is False.
    """
    return subgraph_edge_count(tree.graph, bind=bind)


def get_subtree_vertex_count(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Count nodes inside the subtree of every node, including that node.

    The root's value is the number of nodes in the tree. A leaf's value is 1.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_vertex_count`` and return None. By default False.

    Returns
    -------
    ndarray or None
        One count per node when ``bind`` is False.
    """
    return subgraph_vertex_count(tree.graph, bind=bind)


def get_subtree_branch_count(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Count branching nodes inside the subtree of every node.

    A branch has at least two children. The subtree root is included when it
    is a branch.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_branch_count`` and return None. By default False.

    Returns
    -------
    ndarray or None
        One count per node when ``bind`` is False.
    """
    return subgraph_branch_count(tree.graph, bind=bind)


enrich_tree_graph_docstrings(globals())
