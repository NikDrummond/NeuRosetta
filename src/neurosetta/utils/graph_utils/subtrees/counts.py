"""Per-vertex subtree counts.

Every count is the downstream subtree rooted at that vertex. The edge that
connects the vertex to its parent is outside the subtree: a leaf has one
vertex, one leaf, zero edges, and zero branches.

For every vertex of a valid tree, ``vertex_count = edge_count + 1``.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import int64, ndarray, ones

from ._accumulate import (
    accumulate_subtree,
    bind_or_return,
    prepare_subtree_traversal,
    zeros_like_vertices,
)


def _count(
    g: Graph,
    *,
    name: str,
    vertex_values: ndarray,
    edge_weights: ndarray | None,
    bind: bool,
) -> ndarray | None:
    traversal = prepare_subtree_traversal(g)
    acc = accumulate_subtree(
        traversal,
        vertex_values=vertex_values,
        edge_weights=edge_weights,
        reduction="sum",
    )
    return bind_or_return(g, name, acc.astype(int64, copy=False), "int64_t", bind)


def subgraph_leaf_count(g: Graph, bind: bool = True) -> ndarray | None:
    """Count descendant leaves in each vertex's subtree.

    A leaf is a vertex with out-degree 0. The subtree root is included, so a
    leaf-rooted subtree has leaf count 1.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. Active graph-tool filters are rejected.
    bind : bool, optional
        If True, store vertex property ``subtree_leaf_count`` and return None.
        If False, return an ``int64`` array and do not modify *g*.
        By default True.

    Returns
    -------
    ndarray or None
        One leaf count per vertex when ``bind`` is False.

    See Also
    --------
    subgraph_edge_count
    subgraph_vertex_count
    subgraph_branch_count
    """
    traversal = prepare_subtree_traversal(g)
    leaves = (traversal.out_degree == 0).astype(int64)
    acc = accumulate_subtree(traversal, vertex_values=leaves, reduction="sum")
    return bind_or_return(g, "subtree_leaf_count", acc, "int64_t", bind)


def subgraph_edge_count(g: Graph, bind: bool = True) -> ndarray | None:
    """Count edges inside each vertex's subtree.

    The incoming edge that joins the subtree root to its parent is excluded.
    A leaf therefore has edge count 0.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_edge_count`` and return None. By default True.

    Returns
    -------
    ndarray or None
        One edge count per vertex when ``bind`` is False.

    See Also
    --------
    subgraph_vertex_count
    subgraph_leaf_count
    """
    traversal = prepare_subtree_traversal(g)
    weights = ones(g.num_edges(), dtype=int64)
    return _count(
        g,
        name="subtree_edge_count",
        vertex_values=zeros_like_vertices(traversal, dtype=int64),
        edge_weights=weights,
        bind=bind,
    )


def subgraph_vertex_count(g: Graph, bind: bool = True) -> ndarray | None:
    """Count vertices inside each subtree, including the subtree root.

    On a valid tree this equals ``subgraph_edge_count + 1`` at every vertex.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_vertex_count`` and return None. By default True.

    Returns
    -------
    ndarray or None
        One vertex count per vertex when ``bind`` is False.
    """
    traversal = prepare_subtree_traversal(g)
    ones_v = ones(len(traversal.children), dtype=int64)
    acc = accumulate_subtree(traversal, vertex_values=ones_v, reduction="sum")
    return bind_or_return(g, "subtree_vertex_count", acc, "int64_t", bind)


def subgraph_branch_count(g: Graph, bind: bool = True) -> ndarray | None:
    """Count branching vertices inside each subtree.

    A branch has out-degree >= 2. The subtree root is included when it is
    itself a branch. Leaves and transitive (in-degree 1, out-degree 1)
    vertices contribute 0.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_branch_count`` and return None. By default True.

    Returns
    -------
    ndarray or None
        One branch count per vertex when ``bind`` is False.
    """
    traversal = prepare_subtree_traversal(g)
    branches = (traversal.out_degree >= 2).astype(int64)
    acc = accumulate_subtree(traversal, vertex_values=branches, reduction="sum")
    return bind_or_return(g, "subtree_branch_count", acc, "int64_t", bind)
