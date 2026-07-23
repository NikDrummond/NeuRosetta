"""Functions for counting nodes in tree graphs."""

from ...core import _Tree
from ...utils.graph_utils import (
    count_roots,
    count_vertices,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_vertices,
)


def count_tree_roots(tree: _Tree) -> int:
    """Count the number of root nodes (those with in-degree = 0).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of root nodes.
    """
    return count_roots(tree.graph)


def count_tree_nodes(tree: _Tree) -> int:
    """Count the number of nodes in the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of nodes in the tree graph.
    """
    return count_vertices(tree.graph)


def count_tree_edges(tree: _Tree) -> int:
    """Count the number of edges in the tree graph.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of edges in the tree.
    """
    return count_edges(tree.graph)


def count_tree_leaves(tree: _Tree) -> int:
    """Count the number of leaf nodes (those with out-degree = 0).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of leaf nodes.
    """
    return count_leaves(tree.graph)


def count_tree_branches(tree: _Tree) -> int:
    """Count the number of branch nodes (those with out-degree > 1).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of branching nodes.
    """
    return count_branches(tree.graph)


def count_tree_transitive_nodes(tree: _Tree) -> int:
    """Count the number of transitive nodes (in-degree == 1 and out-degree == 1).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of transitive vertices.
    """
    return count_transitive_vertices(tree.graph)