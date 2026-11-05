"""Extract the vertex/edge subset marked by a subtree mask."""

from __future__ import annotations

from graph_tool.all import Graph

from ..gt_properties import raise_internal_property_missing, revert_core_properties


def extract_subgraph(g: Graph, revert_properties: bool = True) -> Graph:
    """Extract a subgraph using vertex and edge filters.

    Requires the graph to have ``v_subtree_mask`` (vertex) and
    ``e_subtree_mask`` (edge) properties set as boolean filters.

    Parameters
    ----------
    g : Graph
        Graph with vertex filter ``v_subtree_mask`` and edge filter
        ``e_subtree_mask``.
    revert_properties : bool, optional
        If True, remove non-core properties after extraction. By default True.

    Returns
    -------
    Graph
        The filtered and purged graph (modified in place and returned).
    """
    raise_internal_property_missing(g, "v_subtree_mask", "v")
    raise_internal_property_missing(g, "e_subtree_mask", "e")

    g.set_vertex_filter(g.vp["v_subtree_mask"])
    g.set_edge_filter(g.ep["e_subtree_mask"])

    g.purge_vertices()
    g.purge_edges()

    if revert_properties:
        revert_core_properties(g)

    return g
