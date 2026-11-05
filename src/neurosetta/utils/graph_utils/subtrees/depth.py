"""Downstream subtree depth and leaf distance.

These are not :func:`neurosetta.utils.graph_utils` node depth. Node depth
counts edges from the root *down to* a vertex. Subtree depth counts edges
from the vertex *down to* a descendant leaf.

``subgraph_branch_depth`` excludes the starting vertex. A branch whose
children are all leaves has branch depth 0. Each downstream branch on the
longest path adds one.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import float64, int64, ndarray, ones, zeros

from ._accumulate import (
    accumulate_subtree,
    bind_or_return,
    length_metric_name,
    prepare_subtree_traversal,
    resolve_edge_weights,
    zeros_like_vertices,
)


def subgraph_depth(g: Graph, bind: bool = True) -> ndarray | None:
    """Maximum number of downstream edges from each vertex to a leaf.

    A leaf is 0. A vertex whose farthest leaf is its child is 1. This does
    not use coordinates or edge lengths.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_depth`` and return None. By default True.

    Returns
    -------
    ndarray or None
        ``int64`` depth per vertex when ``bind`` is False.

    See Also
    --------
    subgraph_branch_depth
    subgraph_height
    """
    traversal = prepare_subtree_traversal(g)
    acc = accumulate_subtree(
        traversal,
        vertex_values=zeros_like_vertices(traversal, dtype=int64),
        edge_weights=ones(g.num_edges(), dtype=int64),
        reduction="max",
    )
    return bind_or_return(g, "subtree_depth", acc, "int64_t", bind)


def subgraph_branch_depth(g: Graph, bind: bool = True) -> ndarray | None:
    """Maximum number of downstream branching vertices on a path to a leaf.

    The starting vertex is **not** included, even when it is a branch.
    A branch is a vertex with out-degree >= 2. Leaves and chains of
    transitive vertices contribute 0 until a downstream branch is reached.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_branch_depth`` and return None.
        By default True.

    Returns
    -------
    ndarray or None
        ``int64`` branch depth per vertex when ``bind`` is False.

    Notes
    -----
    This is not Strahler order and not the number of branches between the
    root and the vertex. Strahler order is a hierarchy label; this is a
    downstream path count that skips the start vertex.

    Examples
    --------
    On ``0 → {1, 2}``, ``2 → {3, 4}`` with 1, 3, and 4 leaves:

    * leaves have branch depth 0
    * vertex 2 has branch depth 0 (its children are not branches)
    * vertex 0 has branch depth 1 (vertex 2 is a downstream branch)
    """
    traversal = prepare_subtree_traversal(g)
    is_branch = traversal.out_degree >= 2
    weights = zeros(g.num_edges(), dtype=int64)
    for vertex, kids in enumerate(traversal.children):
        for i, child in enumerate(kids):
            if is_branch[child]:
                weights[traversal.edge_index[vertex][i]] = 1
    acc = accumulate_subtree(
        traversal,
        vertex_values=zeros_like_vertices(traversal, dtype=int64),
        edge_weights=weights,
        reduction="max",
    )
    return bind_or_return(g, "subtree_branch_depth", acc, "int64_t", bind)


def subgraph_height(
    g: Graph,
    bind: bool = True,
    length_property: str = "Path_length",
    weights: ndarray | None = None,
) -> ndarray | None:
    """Maximum cable distance from each vertex to a descendant leaf.

    Edge weights are summed along the downstream path. The incoming edge of
    the vertex is excluded. A leaf has height 0.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store ``subtree_height`` or ``subtree_height_euclidean``.
        By default True.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge weights used when *weights* is None. By default ``"Path_length"``.
        Physical cable is not replaced by Euclidean length.
    weights : ndarray or None, optional
        Explicit nonnegative weights in edge-index order.

    Returns
    -------
    ndarray or None
        ``float64`` height per vertex when ``bind`` is False.
    """
    traversal = prepare_subtree_traversal(g)
    resolved = resolve_edge_weights(g, length_property, weights)
    acc = accumulate_subtree(
        traversal,
        vertex_values=zeros_like_vertices(traversal),
        edge_weights=resolved,
        reduction="max",
    )
    name = length_metric_name("subtree_height", length_property)
    return bind_or_return(g, name, acc, "double", bind)


def subgraph_mean_leaf_distance(
    g: Graph,
    bind: bool = True,
    length_property: str = "Path_length",
    weights: ndarray | None = None,
) -> ndarray | None:
    """Mean cable distance from each vertex to its descendant leaves.

    For a vertex ``v`` with descendant leaves ``L(v)``,

    .. math::

        \\bar{d}(v) = \\frac{1}{|L(v)|} \\sum_{\\ell \\in L(v)} d(v, \\ell)

    where ``d(v, ℓ)`` is the sum of edge weights on the unique directed path.
    A leaf has mean 0. The parent edge of ``v`` is not included.

    The sum is built in one reverse-topological pass:

    ``dist_sum(v) = Σ_c (leaf_count(c) * w(v, c) + dist_sum(c))``.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. Every vertex reaches at least one leaf, so the
        denominator is at least 1.
    bind : bool, optional
        If True, store ``subtree_mean_leaf_distance`` or
        ``subtree_mean_leaf_distance_euclidean``. By default True.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge weights used when *weights* is None. By default ``"Path_length"``.
    weights : ndarray or None, optional
        Explicit nonnegative weights in edge-index order.

    Returns
    -------
    ndarray or None
        ``float64`` mean distance per vertex when ``bind`` is False.
    """
    traversal = prepare_subtree_traversal(g)
    resolved = resolve_edge_weights(g, length_property, weights)
    leaves = (traversal.out_degree == 0).astype(float64)
    leaf_count = accumulate_subtree(traversal, vertex_values=leaves, reduction="sum")
    dist_sum = zeros(leaf_count.shape[0], dtype=float64)
    for vertex in traversal.order:
        kids = traversal.children[int(vertex)]
        if not kids:
            continue
        total = 0.0
        eidx = traversal.edge_index[int(vertex)]
        for i, child in enumerate(kids):
            total += float(leaf_count[child]) * float(resolved[eidx[i]]) + float(dist_sum[child])
        dist_sum[int(vertex)] = total
    mean = zeros(leaf_count.shape[0], dtype=float64)
    reachable = leaf_count > 0
    mean[reachable] = dist_sum[reachable] / leaf_count[reachable]
    name = length_metric_name("subtree_mean_leaf_distance", length_property)
    return bind_or_return(g, name, mean, "double", bind)
