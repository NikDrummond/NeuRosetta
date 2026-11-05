"""Ordered vertex and edge paths on a rooted tree.

Edges point from parent to child. ``directed=True`` follows that
orientation, so a path exists only from an ancestor to a descendant.
``directed=False`` walks the unique underlying path and does not reverse
the stored graph. ``graph_tool.topology.shortest_path`` has no ``directed``
argument, so direction-independent retrieval uses
``GraphView(g, directed=False)``.

Active graph-tool filters are rejected. A filter keeps the original
property index space while hiding vertices, and path arrays would no
longer line up with those properties.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_tool.all import Graph, GraphView
from graph_tool.topology import shortest_distance, shortest_path
from numpy import column_stack, int64, ndarray, zeros

from ..tree_validation import as_vertex_index, validate_rooted_tree


@dataclass(slots=True)
class _PathHit:
    """One retrieved path. Edge descriptors are the original graph edges."""

    vertices: ndarray
    edges: list
    pairs: ndarray


def _retrieve(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool,
) -> _PathHit | None:
    """Return the unique path, or None when the direction forbids it."""
    validate_rooted_tree(g)
    src = as_vertex_index(g, source)
    tgt = as_vertex_index(g, target)
    if src == tgt:
        return _PathHit(
            vertices=array_one(src),
            edges=[],
            pairs=zeros((0, 2), dtype=int64),
        )

    graph = g if directed else GraphView(g, directed=False)
    found_vertices, found_edges = shortest_path(graph, graph.vertex(src), graph.vertex(tgt))
    if not found_vertices:
        return None
    vertices = zeros(len(found_vertices), dtype=int64)
    for i, vertex in enumerate(found_vertices):
        vertices[i] = int(vertex)
    if vertices.size == 1:
        pairs = zeros((0, 2), dtype=int64)
    else:
        pairs = column_stack((vertices[:-1], vertices[1:])).astype(int64, copy=False)
    return _PathHit(vertices=vertices, edges=list(found_edges), pairs=pairs)


def array_one(index: int) -> ndarray:
    """Return a length-1 int64 index array."""
    out = zeros(1, dtype=int64)
    out[0] = index
    return out


def path_vertices(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return ordered vertex indices from ``source`` to ``target``.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. One root, one parent for every other vertex,
        no directed cycle.
    source, target : int
        Vertex indices. ``source == target`` is the one-vertex path.
    directed : bool, optional
        Follow stored edge direction when True. When False, use the unique
        undirected path. The graph is not rewritten. By default True.

    Returns
    -------
    ndarray or None
        int64 indices, including both endpoints. ``None`` when no path
        exists under ``directed``.

    Raises
    ------
    TypeError
        If an index is not an integer. Booleans are rejected.
    ValueError
        If the graph is not a rooted tree, a filter is active, or an index
        is outside the vertex range.

    Examples
    --------
    On a tree with edges ``1 -> 3`` and ``1 -> 4``, ``path_vertices(g, 3, 4)``
    is ``None``. ``path_vertices(g, 3, 4, directed=False)`` is
    ``[3, 1, 4]``.

    See Also
    --------
    path_edges : endpoint rows for the same path
    path_exists : existence without building the vertex array
    """
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    return found.vertices


def path_edges(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return traversal-ordered edge endpoints of shape ``(n_edges, 2)``.

    Each row is ``(upstream, downstream)`` in the direction of the walk.
    That can disagree with the stored parent-to-child orientation. Property
    lookup still uses the original edge, so a reversed walk reads the same
    weight as the stored edge.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        int64 endpoint pairs. Shape ``(0, 2)`` when ``source == target``.
        ``None`` when the path does not exist.

    Examples
    --------
    Stored edges ``1 -> 3`` and ``1 -> 4`` give

    ``path_edges(g, 3, 4, directed=False) == [[3, 1], [1, 4]]``.

    See Also
    --------
    path_vertices
    path_edge_values : property values in this same order
    """
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    return found.pairs


def path_exists(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> bool:
    """Return whether ``target`` is reachable from ``source``.

    Uses :func:`graph_tool.topology.shortest_distance` (hop count). A
    finite hop count on a tree is at most ``num_vertices - 1``. The
    unreachable sentinel is larger than that, so the vertex array is not
    built.

    ``source == target`` is True. On a validated tree every pair is
    reachable when ``directed`` is False, because the underlying path is
    unique.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Vertex indices.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    bool
        True when a path exists under ``directed``.

    See Also
    --------
    is_ancestor : directed reachability expressed as ancestry
    path_vertices
    """
    validate_rooted_tree(g)
    src = as_vertex_index(g, source)
    tgt = as_vertex_index(g, target)
    if src == tgt:
        return True
    hops = int(shortest_distance(g, src, target=tgt, directed=directed))
    return hops < int(g.num_vertices())
