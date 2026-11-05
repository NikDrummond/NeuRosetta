"""Root distances for Sholl analysis.

Cumulative modes reuse the unique directed path from the root. When the edge
property is already on the graph, :func:`root_distance` (graph-tool
``shortest_distance``) supplies it and the map is **not** written onto the
graph. Callers that pass an explicit weight array get the same sum from the
reverse-topological adjacency, so a temporary property is never created and
``Root_distance`` cannot be confused with another metric.

Radial distance is not computed here. It is the straight-line distance from a
center to a point in space, which is not a path sum.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import array, float64, ndarray, zeros

from ..paths import root_distance
from ..subtrees._accumulate import SubtreeTraversal, resolve_edge_weights


def cumulative_root_distance(
    g: Graph,
    traversal: SubtreeTraversal,
    length_property: str,
    weights: ndarray | None = None,
) -> ndarray:
    """Return the weighted distance from the root to every vertex.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    traversal : SubtreeTraversal
        Child lists for the explicit-weight path. Ignored when *weights* is
        None and :func:`root_distance` is used instead.
    length_property : str
        ``Path_length`` or ``Euclidean_length``.
    weights : ndarray or None, optional
        Nonnegative edge weights in edge-index order. ``None`` reads
        *length_property* from *g*.

    Returns
    -------
    ndarray
        ``float64`` distance per vertex. The root is 0.

    Notes
    -----
    ``path_length`` mode must use physical ``Path_length``. This function
    does not invent that property from coordinates.
    """
    if weights is None:
        # Validate before shortest_distance. The returned map is not stored
        # on the graph, so it cannot overwrite ``Root_distance``.
        resolve_edge_weights(g, length_property, None, positive=False)
        dist_map = root_distance(g, length_property)
        return array(dist_map.a, dtype=float64, copy=True)

    resolved = resolve_edge_weights(g, length_property, weights, positive=False)
    dist = zeros(len(traversal.children), dtype=float64)
    # ``traversal.order`` is children before parents. Walk it backwards so
    # each parent is final before its children.
    for vertex in traversal.order[::-1]:
        vertex = int(vertex)
        kids = traversal.children[vertex]
        eidx = traversal.edge_index[vertex]
        base = dist[vertex]
        for i, child in enumerate(kids):
            dist[child] = base + float(resolved[eidx[i]])
    return dist
