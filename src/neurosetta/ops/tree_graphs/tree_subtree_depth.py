"""Downstream subtree depth and leaf distance on a NeuRosetta tree.

``get_node_depth`` counts edges from the root down to a node. These functions
look the other way: from the node down to its descendant leaves.

``get_subtree_branch_depth`` does **not** count the starting node. A branch
whose children are all leaves has branch depth 0.
"""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    subgraph_branch_depth,
    subgraph_depth,
    subgraph_height,
    subgraph_mean_leaf_distance,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_subtree_common import resolve_tree_edge_weights


def get_subtree_depth(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Maximum number of downstream edges from each node to a leaf.

    A leaf is 0. This is a topological count: coordinates and edge lengths
    are not used. It is not :func:`get_node_depth`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_depth`` and return None. By default False.

    Returns
    -------
    ndarray or None
        Int64 depth per node when ``bind`` is False.
    """
    return subgraph_depth(tree.graph, bind=bind)


def get_subtree_branch_depth(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Maximum number of downstream branches on a path to a leaf.

    The starting node is excluded. A branch is a node with two or more
    children. This is not Strahler order and not the branch order measured
    from the root.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_branch_depth`` and return None.
        By default False.

    Returns
    -------
    ndarray or None
        Int64 branch depth per node when ``bind`` is False.
    """
    return subgraph_branch_depth(tree.graph, bind=bind)


def get_subtree_height(
    tree: _Tree,
    bind: bool = False,
    length_property: str = "Path_length",
) -> ndarray | None:
    """Maximum cable distance from each node to a descendant leaf.

    The parent edge is excluded. A leaf has height 0.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_height`` or ``subtree_height_euclidean``
        and return None. By default False.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge weights. By default ``"Path_length"``. A reduced tree missing
        ``Path_length`` raises instead of using chord lengths.

    Returns
    -------
    ndarray or None
        Float height per node when ``bind`` is False.
    """
    weights = resolve_tree_edge_weights(tree, length_property)
    return subgraph_height(
        tree.graph,
        bind=bind,
        length_property=length_property,
        weights=weights,
    )


def get_subtree_mean_leaf_distance(
    tree: _Tree,
    bind: bool = False,
    length_property: str = "Path_length",
) -> ndarray | None:
    """Mean cable distance from each node to its descendant leaves.

    A leaf has mean 0. The parent edge is excluded. The value is the mean
    over leaves, not over nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_mean_leaf_distance`` or
        ``subtree_mean_leaf_distance_euclidean`` and return None.
        By default False.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge weights. By default ``"Path_length"``.

    Returns
    -------
    ndarray or None
        Float mean distance per node when ``bind`` is False.
    """
    weights = resolve_tree_edge_weights(tree, length_property)
    return subgraph_mean_leaf_distance(
        tree.graph,
        bind=bind,
        length_property=length_property,
        weights=weights,
    )


enrich_tree_graph_docstrings(globals())
