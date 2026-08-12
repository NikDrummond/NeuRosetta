"""Functions for counting nodes of different types in a graph."""
from __future__ import annotations

from numpy import where, hstack
from graph_tool.all import Graph

from .vertex_inds import leaf_indices, branch_indices, bifurcation_indices


def count_roots(g: Graph) -> int:
    """Count the number of root nodes in a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Number of root vertices (in-degree == 0).
    """
    return len(where(g.degree_property_map("in").a == 0))


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
    return int(
        sum(
            (g.degree_property_map("out").a == 1)
            & (g.degree_property_map("in").a == 1)
        )
    )

def count_sections(g:Graph)-> int:
    """Count the number of sections in a graph.

    Sections are edges between the root, branch, and leaf nodes

    Parameters
    ----------
    g : Graph
        Directed tree graph

    Returns
    -------
    int
        Number of sections
    """
    b = branch_indices(g)
    l = leaf_indices(g)
    r = where(g.degree_property_map("in").a == 0)[0][0]
    return hstack([b[b != r], l]).shape[0]

def count_bifurcations(g:Graph, include_root:bool = False)-> int:
    """Count the number of bifurcations within a graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph
    include_root : bool, optional
        If True, include root index in count if it is a bifurcation, by default False

    Returns
    -------
    int
        Number of bifurcations
    """
    return len(bifurcation_indices(g, include_root=include_root))


