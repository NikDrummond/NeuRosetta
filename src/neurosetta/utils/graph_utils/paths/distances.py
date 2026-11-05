"""Weighted path lengths and source distances.

Individual paths are the unique tree path from :func:`path_vertices`.
Their length is the sum of the selected edge property, including when the
walk runs against the stored direction. The weight is a scalar on the
edge, so reversing the walk does not change it.

Distances from one source to many vertices use
:func:`graph_tool.topology.shortest_distance`. Unreachable vertices are
``inf``. A temporary double weight map is used when the stored property
is an integer type, because an integer distance map encodes "unreachable"
as the type maximum rather than infinity. Named properties are not
changed, and results are not written onto ``g.vp``.

:func:`root_distance` keeps its historical contract: the default property
is ``Path_length`` with no fallback, and a missing property raises
``_InternalPropertyMissingError``.
"""

from __future__ import annotations

from graph_tool import EdgePropertyMap
from graph_tool.all import Graph, VertexPropertyMap
from graph_tool.topology import shortest_distance
from numpy import asarray, cumsum, empty, float64, inf, ndarray, zeros

from ..gt_properties import raise_internal_property_missing
from ..tree_validation import as_vertex_index, validate_rooted_tree
from ..vertex_inds import root_index
from ._weights import as_double_weights, resolve_edge_property
from .ancestry import lca_from_tables, parent_depth_tables
from .retrieval import _retrieve


def path_length(
    g: Graph,
    source: int,
    target: int,
    ep: str | EdgePropertyMap | None = None,
    *,
    directed: bool = True,
) -> float | None:
    """Return the sum of edge weights along the path.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints. The length is 0 when they are the same vertex.
    ep : str or EdgePropertyMap or None, optional
        Weight property. ``None`` uses ``Path_length``, then
        ``Euclidean_length``. An explicit missing name is an error.
        By default None.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    float or None
        Path weight. ``None`` when no path exists under ``directed``.

    Raises
    ------
    ValueError
        If the requested weights are missing, non-finite, or negative.
    TypeError
        If the property is not a numeric scalar.

    See Also
    --------
    path_cumulative_length
    source_distances
    root_distance
    """
    weights = resolve_edge_property(g, ep, nonnegative=True)
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    total = 0.0
    for edge in found.edges:
        total += float(weights[edge])
    return total


def path_edge_count(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> int | None:
    """Return the number of edges on the path.

    No edge property is required. The count is 0 when ``source == target``.

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
    int or None
        Edge count, or ``None`` when the path does not exist.

    See Also
    --------
    path_edges
    path_length
    """
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    return int(found.pairs.shape[0])


def path_cumulative_length(
    g: Graph,
    source: int,
    target: int,
    ep: str | EdgePropertyMap | None = None,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return the running weight from ``source`` to each path vertex.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    ep : str or EdgePropertyMap or None, optional
        Weight property. Same selection rules as :func:`path_length`.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        float64 array with one entry per path vertex. The first value is
        0. The last value equals :func:`path_length` for the same query.
        ``None`` when the path does not exist.

    Examples
    --------
    Vertices ``[3, 1, 0, 2]`` with step weights 4, 2, and 3 produce
    ``[0.0, 4.0, 6.0, 9.0]``.

    See Also
    --------
    path_length
    path_vertices
    """
    weights = resolve_edge_property(g, ep, nonnegative=True)
    found = _retrieve(g, source, target, directed=directed)
    if found is None:
        return None
    steps = empty(len(found.edges), dtype=float64)
    for i, edge in enumerate(found.edges):
        steps[i] = float(weights[edge])
    out = empty(found.vertices.size, dtype=float64)
    out[0] = 0.0
    if steps.size:
        out[1:] = cumsum(steps)
    return out


def source_distances(
    g: Graph,
    source: int,
    ep: str | EdgePropertyMap | None = None,
    *,
    directed: bool = True,
) -> VertexPropertyMap:
    """Return weighted distances from ``source`` to every vertex.

    Uses :func:`graph_tool.topology.shortest_distance`. On a tree the
    result is the unique-path distance. Vertices that cannot be reached
    under ``directed`` are ``inf``, which is distinct from a real
    zero-length path at the source.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source : int
        Source vertex. It does not have to be the root.
    ep : str or EdgePropertyMap or None, optional
        Weight property. ``None`` uses ``Path_length``, then
        ``Euclidean_length``.
    directed : bool, optional
        Stored direction when True. ``shortest_distance`` receives this
        flag directly. By default True.

    Returns
    -------
    VertexPropertyMap
        Double distances. Not inserted into ``g.vp``. Distance at
        ``source`` is 0.

    See Also
    --------
    root_distance
    path_lengths_to_vertices
    """
    validate_rooted_tree(g)
    src = as_vertex_index(g, source)
    weights = as_double_weights(g, resolve_edge_property(g, ep, nonnegative=True))
    return shortest_distance(g, src, weights=weights, directed=directed)


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
        ``"Path_length"``. There is no fallback to another property.

    Returns
    -------
    VertexPropertyMap
        Per-vertex distance from the root under the given edge weights.
        Root distance is ``0``. The map is not inserted into ``g``. Callers
        that bind it must use a name that records which edge property was
        used (``Root_distance`` for ``Path_length``,
        ``Root_euclidean_distance`` for ``Euclidean_length``).

    Raises
    ------
    _InternalPropertyMissingError
        If the edge property ``ep`` is missing from ``g``.
    """
    raise_internal_property_missing(g, ep, "e")
    source = root_index(g)
    return shortest_distance(g=g, source=source, weights=g.ep[ep])


def path_lengths_to_vertices(
    g: Graph,
    source: int,
    targets,
    ep: str | EdgePropertyMap | None = None,
    *,
    directed: bool = True,
) -> ndarray:
    """Return distances from one source to each target.

    One :func:`graph_tool.topology.shortest_distance` call evaluates every
    target. Order matches ``targets``. Unreachable targets are ``inf``.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source : int
        Source vertex.
    targets : sequence of int
        Target indices. An empty sequence returns a length-0 array.
    ep : str or EdgePropertyMap or None, optional
        Weight property. Same rules as :func:`path_length`.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray
        float64 distances, one per target.

    See Also
    --------
    source_distances
    pairwise_path_lengths
    """
    validate_rooted_tree(g)
    src = as_vertex_index(g, source)
    target_list = [as_vertex_index(g, vertex) for vertex in targets]
    if not target_list:
        return zeros(0, dtype=float64)
    weights = as_double_weights(g, resolve_edge_property(g, ep, nonnegative=True))
    queried = [g.vertex(vertex) for vertex in target_list]
    found = shortest_distance(g, src, target=queried, weights=weights, directed=directed)
    return asarray(found, dtype=float64).copy()


def pairwise_path_lengths(
    g: Graph,
    pairs,
    ep: str | EdgePropertyMap | None = None,
    *,
    directed: bool = True,
) -> ndarray:
    """Return one distance per source-target pair.

    Root distances are computed once with
    :func:`graph_tool.topology.shortest_distance`. Each pair is then the
    unique path through the lowest common ancestor:

    * undirected: ``d(u) + d(v) - 2 d(lca)``
    * directed: ``d(v) - d(u)`` when ``u`` is an ancestor of ``v``,
      otherwise ``inf``

    Pairs that share a source do not cause another graph traversal.
    Nothing is cached on the graph.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    pairs : array-like, shape (n_pairs, 2)
        Source, target indices. Empty input returns a length-0 array.
    ep : str or EdgePropertyMap or None, optional
        Nonnegative weight property.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray
        float64 distances in input order. Unreachable pairs are ``inf``.

    Raises
    ------
    ValueError
        If ``pairs`` is not an ``(n, 2)`` index array, or weights are invalid.

    See Also
    --------
    path_lengths_to_vertices
    lowest_common_ancestor
    """
    table = asarray(pairs)
    if table.size == 0:
        return zeros(0, dtype=float64)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ValueError(f"pairs must have shape (n_pairs, 2), got {table.shape}.")
    validate_rooted_tree(g)
    sources = [as_vertex_index(g, value) for value in table[:, 0]]
    targets = [as_vertex_index(g, value) for value in table[:, 1]]
    weights = as_double_weights(g, resolve_edge_property(g, ep, nonnegative=True))
    parents, depth, root = parent_depth_tables(g)
    dist_map = shortest_distance(g, root, weights=weights, directed=True)
    root_dist = asarray(dist_map.a, dtype=float64).copy()

    out = empty(len(sources), dtype=float64)
    for i, (src, tgt) in enumerate(zip(sources, targets, strict=True)):
        if src == tgt:
            out[i] = 0.0
            continue
        common = lca_from_tables(parents, depth, src, tgt)
        if directed and common != src:
            out[i] = inf
        elif directed:
            out[i] = root_dist[tgt] - root_dist[src]
        else:
            out[i] = root_dist[src] + root_dist[tgt] - 2.0 * root_dist[common]
    return out
