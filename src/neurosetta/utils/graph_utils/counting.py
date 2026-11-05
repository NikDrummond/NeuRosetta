"""Functions for counting nodes of different types in a graph."""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import array, count_nonzero, hstack, where

from .vertex_inds import bifurcation_indices, branch_indices, core_indices, leaf_indices


def count_roots(g: Graph) -> int:
    """Count the number of root nodes in a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of visible root vertices (in-degree == 0).

    Notes
    -----
    ``len(numpy.where(...))`` is the dimensionality of the index tuple, not
    the number of matches, so it cannot be used here. An active vertex filter
    is respected: only vertices returned by ``get_vertices()`` are counted.
    """
    vertices = g.get_vertices()
    if len(vertices) == 0:
        return 0
    return int(count_nonzero(g.get_in_degrees(vertices) == 0))


def count_vertices(g: Graph) -> int:
    """Count the number of vertices in a graph.

    Parameters
    ----------
    g : Graph
        Input graph.

    Returns
    -------
    int
        Total number of vertices.
    """
    return g.num_vertices()


def count_edges(g: Graph) -> int:
    """Count the number of edges in a graph.

    Parameters
    ----------
    g : Graph
        Input graph.

    Returns
    -------
    int
        Total number of edges.
    """
    return g.num_edges()


def count_leaves(g: Graph) -> int:
    """Count the number of leaf vertices in a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of leaf vertices (out-degree == 0).
    """
    return len(leaf_indices(g))


def count_branches(g: Graph) -> int:
    """Count the number of branch vertices in a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of branch vertices (out-degree > 1).
    """
    return len(branch_indices(g))


def count_transitive_vertices(g: Graph) -> int:
    """Count the number of transitive vertices in a graph.

    Transitive vertices have in-degree == 1 and out-degree == 1.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of transitive vertices.
    """
    out_map = g.degree_property_map("out")
    in_map = g.degree_property_map("in")
    out_degree = array(out_map.a, copy=True)
    in_degree = array(in_map.a, copy=True)
    return int(((out_degree == 1) & (in_degree == 1)).sum())


def count_sections(g: Graph) -> int:
    """Count the number of sections in a graph.

    Sections are edges between the root, branch, and leaf nodes.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of sections.
    """
    b = branch_indices(g)
    leaves = leaf_indices(g)
    in_map = g.degree_property_map("in")
    r = int(where(array(in_map.a, copy=True) == 0)[0][0])
    return hstack([b[b != r], leaves]).shape[0]


def count_bifurcations(g: Graph, include_root: bool = False) -> int:
    """Count the number of bifurcations within a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    include_root : bool, optional
        If True, include the root in the count when it is a bifurcation.
        By default False.

    Returns
    -------
    int
        Number of bifurcations.
    """
    return len(bifurcation_indices(g, include_root=include_root))


def count_core_vertices(g: Graph, include_root: bool = True) -> int:
    """Count core vertices (root, branches, and leaves).

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    include_root : bool, optional
        If True, include the root in the count. By default True.

    Returns
    -------
    int
        Number of core vertices.
    """
    return len(core_indices(g, include_root=include_root))
