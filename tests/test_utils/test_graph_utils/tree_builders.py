"""Builders for small directed trees with known measurements."""

from __future__ import annotations

import numpy as np
from graph_tool.all import Graph


def make_directed_tree(
    edges,
    *,
    n: int | None = None,
    coords=None,
    path_length=None,
    euclidean=None,
) -> Graph:
    """Directed graph with optional coordinates and edge lengths.

    Edge-length arrays follow edge-index order, which is the order of *edges*
    for a freshly built graph.
    """
    edges = list(edges)
    if n is None:
        n = 1 if not edges else int(max(max(a, b) for a, b in edges) + 1)
    g = Graph(directed=True)
    g.add_vertex(n)
    if edges:
        g.add_edge_list(np.asarray(edges, dtype=int))
    if coords is not None:
        xyz = np.asarray(coords, dtype=float)
        g.vp["x"] = g.new_vp("double", xyz[:, 0])
        g.vp["y"] = g.new_vp("double", xyz[:, 1])
        g.vp["z"] = g.new_vp("double", xyz[:, 2])
    if path_length is not None:
        g.ep["Path_length"] = g.new_ep("double", np.asarray(path_length, dtype=float))
    if euclidean is not None:
        g.ep["Euclidean_length"] = g.new_ep("double", np.asarray(euclidean, dtype=float))
    return g
