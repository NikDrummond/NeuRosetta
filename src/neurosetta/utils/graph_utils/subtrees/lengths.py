"""Subtree cable length.

``subgraph_path_length`` is the sum of edge weights strictly inside the
subtree. The edge that enters the subtree root from its parent is excluded,
so a leaf has cable length 0 and the root's value is the total cable of the
tree.

``Path_length`` is physical cable, including lengths of sections that were
collapsed by reduction. ``Euclidean_length`` is the straight-line length of
the edge as stored. This function does not replace one with the other. Pass
``weights`` only when the caller has already chosen the array explicitly.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import ndarray

from ._accumulate import (
    accumulate_subtree,
    bind_or_return,
    length_metric_name,
    prepare_subtree_traversal,
    resolve_edge_weights,
    zeros_like_vertices,
)


def subgraph_path_length(
    g: Graph,
    bind: bool = True,
    length_property: str = "Path_length",
    weights: ndarray | None = None,
) -> ndarray | None:
    """Sum edge lengths inside each vertex's subtree.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    bind : bool, optional
        If True, store a vertex property and return None. The property is
        ``subtree_path_length`` for ``Path_length`` and
        ``subtree_path_length_euclidean`` for ``Euclidean_length``.
        By default True.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge property summed when *weights* is None. By default
        ``"Path_length"``. Missing properties are an error; Euclidean length
        is not substituted for physical cable.
    weights : ndarray or None, optional
        Explicit nonnegative edge weights in graph-tool edge-index order.
        When given, *length_property* is only used to name the bound property.

    Returns
    -------
    ndarray or None
        ``float64`` cable length per vertex when ``bind`` is False. Leaves
        are 0.

    Notes
    -----
    Zero-length edges are allowed. Negative or non-finite weights are not.
    The graph is not modified when ``bind`` is False.

    See Also
    --------
    subgraph_height
    subgraph_mean_leaf_distance
    """
    traversal = prepare_subtree_traversal(g)
    resolved = resolve_edge_weights(g, length_property, weights, positive=False)
    acc = accumulate_subtree(
        traversal,
        vertex_values=zeros_like_vertices(traversal),
        edge_weights=resolved,
        reduction="sum",
    )
    name = length_metric_name("subtree_path_length", length_property)
    return bind_or_return(g, name, acc, "double", bind)
