"""Subtree score used to pick a representative subtree root.

For a branching vertex (out-degree >= 2)

.. math::

    s(v) = 1 - \\frac{C(v)}{C} + \\frac{L(v)}{L}

where ``C(v)`` is subtree cable, ``C`` is total cable, ``L(v)`` is the
subtree leaf count, and ``L`` is the total leaf count. Every other vertex
scores 0.

Cable prefers the ``Path_length`` edge property and falls back to
``Euclidean_length`` when ``Path_length`` is absent. That fallback is kept
for compatibility with existing ``subgraph_score`` callers. New cable
metrics do not use it.

If total cable is 0, the cable term is defined as 0 (there is no cable to
apportion) and the score is the leaf fraction. Previously a ``0 / 0`` went
through ``nan_to_num`` and wiped the leaf term as well.
"""

from __future__ import annotations

from graph_tool.all import Graph
from graph_tool.topology import topological_sort
from numpy import argmax, array, asarray, float64, ndarray, zeros

from ..gt_properties import g_has_property, raise_internal_property_missing


def _cable_length_property(g: Graph):
    """Return the edge property map used as physical cable length.

    Prefers ``Path_length``; falls back to ``Euclidean_length``.
    """
    if g_has_property(g, "Path_length", "e"):
        return g.ep["Path_length"]
    if g_has_property(g, "Euclidean_length", "e"):
        return g.ep["Euclidean_length"]
    raise_internal_property_missing(g, "Path_length", "e")


def subgraph_score(g: Graph, bind: bool = True) -> None | ndarray:
    """Compute subgraph score for each vertex in a directed tree.

    The score combines two terms for branch points (out-degree >= 2):

    * Cable fraction: ``1 - (subtree_cable / total_cable)``
    * Leaf fraction: ``subtree_leaves / total_leaves``

    Non-branch vertices receive a score of 0.

    Parameters
    ----------
    g : Graph
        Directed tree graph with edge property ``Path_length`` or
        ``Euclidean_length``.
    bind : bool, optional
        If True, bind the result as vertex property ``subgraph_score`` and
        return None. If False, return the score array. By default True.

    Returns
    -------
    None | ndarray
        If bind=False, returns array of scores with shape ``(n_vertices,)``.
        Otherwise returns None and adds the ``subgraph_score`` vertex property.

    Notes
    -----
    Subtree cable excludes the incoming edge of the vertex, matching the
    historical implementation. Child contributions are gathered from an
    adjacency list, so each edge is read once.

    A zero total cable length contributes 0 to the cable term instead of
    propagating NaN. An empty graph, or a graph with no leaves, raises
    ``ValueError``.
    """
    n = int(g.num_vertices())
    if n == 0:
        raise ValueError("Cannot score an empty graph.")

    path_length = _cable_length_property(g)
    total_cable = float(asarray(path_length.a, dtype=float64).sum())
    out_map = g.degree_property_map("out")
    out_degrees = array(out_map.a, copy=True)
    is_leaf = out_degrees == 0
    total_leaves = int(is_leaf.sum())
    if total_leaves == 0:
        raise ValueError("Graph has no leaves; subgraph_score expects a directed tree.")

    children: list[list[int]] = [[] for _ in range(n)]
    edge_length: list[list[float]] = [[] for _ in range(n)]
    for edge in g.edges():
        src = int(edge.source())
        tgt = int(edge.target())
        children[src].append(tgt)
        edge_length[src].append(float(path_length[edge]))

    subtree_cable = zeros(n, dtype=float64)
    subtree_leaves = is_leaf.astype(float64)
    for vertex in reversed(asarray(topological_sort(g), dtype=int)):
        kids = children[int(vertex)]
        if not kids:
            continue
        subtree_cable[int(vertex)] = float(sum(edge_length[int(vertex)])) + float(
            subtree_cable[kids].sum()
        )
        subtree_leaves[int(vertex)] += float(subtree_leaves[kids].sum())

    branch_mask = out_degrees >= 2
    cable_term = zeros(n, dtype=float64)
    if total_cable > 0.0:
        cable_term = 1.0 - (subtree_cable / total_cable)
    score = zeros(n, dtype=float64)
    score[branch_mask] = cable_term[branch_mask] + (
        subtree_leaves[branch_mask] / float(total_leaves)
    )

    if bind:
        g.vp["subgraph_score"] = g.new_vp("double", score)
        return None
    return score


def max_subgraph_ind(g: Graph) -> int:
    """Return the vertex index with the maximum subgraph score.

    Parameters
    ----------
    g : Graph
        Directed tree graph. If ``subgraph_score`` exists it is used.
        Otherwise the score is computed internally without binding.

    Returns
    -------
    int
        Vertex index with the maximum subgraph score.
    """
    if g_has_property(g, "subgraph_score", "v"):
        return int(argmax(g.vp["subgraph_score"].a))
    return int(argmax(subgraph_score(g, bind=False)))
