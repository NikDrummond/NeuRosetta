"""Bottom-up subtree aggregation.

One reverse-topological pass, O(V + E). Callers choose the vertex and edge
contributions. This module does not know about leaves, cable, or Strahler
order.

The input graph is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from graph_tool.all import Graph
from graph_tool.topology import topological_sort
from numpy import array, asarray, float64, int64, isfinite, ndarray, zeros

from ..tree_validation import validate_rooted_tree

Reduction = Literal["sum", "max"]

_LENGTH_SUFFIX = {
    "Path_length": "",
    "Euclidean_length": "_euclidean",
}


@dataclass(slots=True)
class SubtreeTraversal:
    """Child lists and a children-before-parents vertex order.

    ``children[v]`` and ``edge_index[v]`` are parallel. ``edge_index`` values
    index edge-property ``.a`` arrays (graph-tool edge indices).
    """

    root: int
    order: ndarray
    children: list[list[int]]
    edge_index: list[list[int]]
    out_degree: ndarray


def length_metric_name(base: str, length_property: str) -> str:
    """Vertex-property name for a cable metric.

    ``Path_length`` keeps *base* (``subtree_path_length``).
    ``Euclidean_length`` appends ``_euclidean`` so a cached path-length result
    is not overwritten by a straight-line result.

    Raises
    ------
    ValueError
        If *length_property* is not ``Path_length`` or ``Euclidean_length``.
    """
    try:
        suffix = _LENGTH_SUFFIX[length_property]
    except KeyError as exc:
        raise ValueError(
            f"Unknown length property {length_property!r}. "
            "Expected 'Path_length' or 'Euclidean_length'."
        ) from exc
    return f"{base}{suffix}"


def prepare_subtree_traversal(g: Graph) -> SubtreeTraversal:
    """Validate *g* and build the adjacency used by every subtree metric.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. See :func:`validate_rooted_tree`.

    Returns
    -------
    SubtreeTraversal
        Reverse topological order and outgoing adjacency.
    """
    root = validate_rooted_tree(g)
    n = int(g.num_vertices())
    children: list[list[int]] = [[] for _ in range(n)]
    edge_index: list[list[int]] = [[] for _ in range(n)]
    eindex = g.edge_index
    for edge in g.edges():
        src = int(edge.source())
        tgt = int(edge.target())
        children[src].append(tgt)
        edge_index[src].append(int(eindex[edge]))
    topo = asarray(topological_sort(g), dtype=int64)
    if int(topo.size) != n:
        raise ValueError(
            "Topological traversal did not cover all vertices. "
            "Check that the graph is a valid directed tree."
        )
    out_map = g.degree_property_map("out")
    out_degree = array(out_map.a, dtype=int64, copy=True)
    return SubtreeTraversal(
        root=root,
        order=topo[::-1].copy(),
        children=children,
        edge_index=edge_index,
        out_degree=out_degree,
    )


def accumulate_subtree(
    traversal: SubtreeTraversal,
    *,
    vertex_values: ndarray,
    edge_weights: ndarray | None = None,
    reduction: Reduction = "sum",
) -> ndarray:
    """Accumulate one value per vertex from children to parents.

    For every vertex ``v`` with children ``c_i`` along edges ``e_i``:

    * ``sum``: ``vertex_values[v] + Σ_i (w(e_i) + acc[c_i])``
    * ``max``: ``vertex_values[v] + max_i (w(e_i) + acc[c_i])``

    A vertex with no children keeps ``vertex_values[v]``. Missing
    *edge_weights* contribute 0. The incoming edge of ``v`` is not added:
    only edges that start at ``v`` are.

    Parameters
    ----------
    traversal : SubtreeTraversal
        From :func:`prepare_subtree_traversal`.
    vertex_values : ndarray
        One initial contribution per vertex, indexed by vertex id.
    edge_weights : ndarray or None, optional
        One weight per edge, indexed by graph-tool edge index. By default
        None (every edge contributes 0, but children are still combined).
    reduction : {"sum", "max"}, optional
        How child contributions combine. By default ``"sum"``.

    Returns
    -------
    ndarray
        One accumulated value per vertex. Integer inputs stay ``int64``;
        floating inputs stay ``float64``.

    Notes
    -----
    Time is O(V + E). The walk is iterative. *traversal* and the input graph
    are not modified.
    """
    if reduction not in ("sum", "max"):
        raise ValueError(f"reduction must be 'sum' or 'max', got {reduction!r}.")
    values = asarray(vertex_values)
    if values.shape != (len(traversal.children),):
        raise ValueError(f"vertex_values must have one entry per vertex, got shape {values.shape}.")
    use_float = float64 in (values.dtype, getattr(edge_weights, "dtype", values.dtype))
    if edge_weights is not None and edge_weights.dtype.kind == "f":
        use_float = True
    dtype = float64 if use_float else int64
    acc = array(values, dtype=dtype, copy=True)
    weights = None if edge_weights is None else asarray(edge_weights)
    children = traversal.children
    edge_index = traversal.edge_index

    for vertex in traversal.order:
        kids = children[int(vertex)]
        if not kids:
            continue
        eidx = edge_index[int(vertex)]
        if reduction == "sum":
            total = acc[int(vertex)]
            # ``acc[v]`` already holds the vertex contribution.
            for i, child in enumerate(kids):
                extra = 0 if weights is None else weights[eidx[i]]
                total = total + extra + acc[child]
            acc[int(vertex)] = total
            continue
        child0 = kids[0]
        extra0 = 0 if weights is None else weights[eidx[0]]
        best = extra0 + acc[child0]
        for i, child in enumerate(kids[1:], start=1):
            extra = 0 if weights is None else weights[eidx[i]]
            candidate = extra + acc[child]
            if candidate > best:
                best = candidate
        acc[int(vertex)] = acc[int(vertex)] + best
    return acc


def resolve_edge_weights(
    g: Graph,
    length_property: str,
    weights: ndarray | None,
    *,
    positive: bool = False,
) -> ndarray:
    """Return finite, nonnegative edge weights aligned with edge indices.

    Parameters
    ----------
    g : Graph
        Directed graph whose edge-index space defines the array layout.
    length_property : str
        Edge property read when *weights* is None. Not silently replaced by
        a different property.
    weights : ndarray or None
        Explicit weights in edge-index order. When given, the property is
        not read.
    positive : bool, optional
        Require every weight to be strictly positive. By default False
        (zero is allowed).

    Returns
    -------
    ndarray
        ``float64`` array of length ``g.num_edges()``.

    Raises
    ------
    ValueError
        If the property is missing, or weights are the wrong shape, non-finite,
        negative, or non-positive when *positive* is True.
    """
    length_metric_name("check", length_property)
    n_edges = int(g.num_edges())
    if weights is None:
        if length_property not in g.ep:
            raise ValueError(
                f"Missing edge property {length_property!r}. "
                "This metric does not substitute a different length definition."
            )
        resolved = asarray(g.ep[length_property].a, dtype=float64)
    else:
        resolved = asarray(weights, dtype=float64)
    if resolved.ndim != 1 or int(resolved.shape[0]) != n_edges:
        raise ValueError(f"Edge weights must have shape ({n_edges},), got {resolved.shape}.")
    if resolved.size and not isfinite(resolved).all():
        raise ValueError(f"Edge weights for {length_property!r} must be finite.")
    if positive and (resolved <= 0).any():
        raise ValueError(f"Edge weights for {length_property!r} must be strictly positive.")
    if (resolved < 0).any():
        raise ValueError(f"Edge weights for {length_property!r} must be nonnegative.")
    return resolved


def bind_or_return(
    g: Graph,
    name: str,
    values: ndarray,
    gt_type: str,
    bind: bool,
) -> ndarray | None:
    """Store *values* as a vertex property, or return them.

    Parameters
    ----------
    g : Graph
        Graph that receives the property when *bind* is True. No other
        properties are written or deleted.
    name : str
        Vertex property name.
    values : ndarray
        One value per vertex.
    gt_type : str
        graph-tool value type, for example ``"double"`` or ``"int64_t"``.
    bind : bool
        If True, bind and return None. If False, return *values* and leave
        *g* unchanged.

    Returns
    -------
    ndarray or None
        *values* when ``bind`` is False, otherwise None.
    """
    if bind:
        g.vp[name] = g.new_vp(gt_type, values)
        return None
    return values


def zeros_like_vertices(traversal: SubtreeTraversal, dtype=float64) -> ndarray:
    """Return a zero array with one entry per vertex."""
    return zeros(len(traversal.children), dtype=dtype)
