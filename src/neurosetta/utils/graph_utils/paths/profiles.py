"""Vertex and edge property values in path order.

Values keep the property's own type. They are not cast to float. A walk
against the stored edge direction still reads the original edge, using the
edge descriptor from :func:`graph_tool.topology.shortest_path` rather than
looking up ``(child, parent)`` in the directed graph.
"""

from __future__ import annotations

from graph_tool import EdgePropertyMap, VertexPropertyMap
from graph_tool.all import Graph
from numpy import array, empty, ndarray

from .retrieval import _retrieve

_DTYPES = {
    "double": float,
    "float": "float32",
    "long double": float,
    "int": "int32",
    "int16_t": "int16",
    "int32_t": "int32",
    "int64_t": "int64",
    "unsigned long": "uint64",
    "bool": bool,
    "string": object,
}


def _vertex_map(g: Graph, vp: str | VertexPropertyMap) -> VertexPropertyMap:
    if isinstance(vp, str):
        if vp not in g.vp:
            raise ValueError(f"Vertex property {vp!r} is not on the graph.")
        return g.vp[vp]
    if isinstance(vp, VertexPropertyMap):
        return vp
    raise TypeError(f"vp must be a property name or VertexPropertyMap, got {type(vp).__name__}.")


def _edge_map(g: Graph, ep: str | EdgePropertyMap) -> EdgePropertyMap:
    if isinstance(ep, str):
        if ep not in g.ep:
            raise ValueError(f"Edge property {ep!r} is not on the graph.")
        return g.ep[ep]
    if isinstance(ep, EdgePropertyMap):
        return ep
    raise TypeError(f"ep must be a property name or EdgePropertyMap, got {type(ep).__name__}.")


def _empty_like(prop) -> ndarray:
    dtype = _DTYPES.get(str(prop.value_type()), object)
    return empty(0, dtype=dtype)


def path_vertex_values(
    g: Graph,
    source: int,
    target: int,
    vp: str | VertexPropertyMap,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return one vertex-property value per path vertex.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    vp : str or VertexPropertyMap
        Property to read.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Values in traversal order. Vector-valued properties keep a trailing
        dimension. ``None`` when the path does not exist.

    See Also
    --------
    path_vertices
    path_edge_values
    """
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    prop = _vertex_map(g, vp)
    values = [prop[g.vertex(int(index))] for index in found.vertices]
    return array(values)


def path_edge_values(
    g: Graph,
    source: int,
    target: int,
    ep: str | EdgePropertyMap,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return one edge-property value per path edge.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    ep : str or EdgePropertyMap
        Property to read. This is not limited to length properties.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Values in traversal order. Shape ``(0,)`` when ``source == target``.
        ``None`` when the path does not exist.

    See Also
    --------
    path_edges
    path_length
    """
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    prop = _edge_map(g, ep)
    if not found.edges:
        return _empty_like(prop)
    values = [prop[edge] for edge in found.edges]
    return array(values)
