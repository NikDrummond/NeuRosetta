"""Functions for counting nodes in tree graphs."""

from ...core import _Tree
from ...utils.graph_utils import (
    count_roots as _count_roots,
    count_vertices as _count_vertices,
    count_edges as _count_edges,
    count_leaves as _count_leaves,
    count_branches as _count_branches,
    count_transitive_vertices as _count_transitive_vertices,
    count_sections as _count_sections,
    count_bifurcations as _count_bifurcations,
)


def count_roots(tree: _Tree) -> int:
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
    return _count_roots(tree.graph)


def count_nodes(tree: _Tree) -> int:
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
    return _count_vertices(tree.graph)


def count_edges(tree: _Tree) -> int:
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
    return _count_edges(tree.graph)


def count_leaves(tree: _Tree) -> int:
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
    return _count_leaves(tree.graph)


def count_branches(tree: _Tree) -> int:
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
    return _count_branches(tree.graph)


def count_transitive_nodes(tree: _Tree) -> int:
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
    return _count_transitive_vertices(tree.graph)

def count_sections(tree: _Tree) -> int:
    """Count the number of sections in a tree.

    Sections are maximal paths between root, branch, and leaf nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Number of sections.
    """
    return _count_sections(tree.graph)

def count_bifurcations(tree: _Tree, include_root: bool = False) -> int:
    """Count the number of bifurcating nodes within a tree

    Parameters
    ----------
    tree : _Tree
        Neuron tree
    include_root : bool, optional
        If True, count the root node if it is a bifurcation, by default False

    Returns
    -------
    int
        Number of bifurcation points within the neuron.
    """
    return _count_bifurcations(tree.graph, include_root=include_root)
