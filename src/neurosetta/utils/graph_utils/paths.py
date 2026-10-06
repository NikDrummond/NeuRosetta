"""Functions for paths / path lengths on graphs."""

from graph_tool.all import Graph, VertexPropertyMap, shortest_distance

from .gt_properties import raise_internal_property_missing
from .vertex_inds import root_index


def root_distance(g: Graph, ep: str = "Path_length") -> VertexPropertyMap:
    """Compute weighted path distance from the root to every vertex.

    Uses :func:`graph_tool.topology.shortest_distance` with the named edge
    property as edge weights. The source is the unique root vertex
    (in-degree == 0).

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    ep : str, optional
        Name of the edge property used as path weights (typically
        ``"Path_length"`` or ``"Euclidean_length"``). By default
        ``"Path_length"``.

    Returns
    -------
    VertexPropertyMap
        Per-vertex distance from the root under the given edge weights.
        Root distance is ``0``.

    Raises
    ------
    AttributeError
        If the edge property ``ep`` is missing from ``g``.
    """
    raise_internal_property_missing(g, ep, "e")
    source = root_index(g)
    return shortest_distance(g=g, source=source, weights=g.ep[ep])
