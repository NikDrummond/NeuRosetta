"""Validation for directed rooted trees.

Subtree metrics, Strahler order, and Sholl analysis all assume one directed
arborescence: a single root, every other vertex has one parent, and there is
no directed cycle. Checks run once per public call, not inside the traversal
loop.

Active graph-tool filters
-------------------------
Vertex and edge filters keep the original property-map index space while hiding
part of the graph. Topological order, ``num_vertices()``, and ``.a`` arrays
then disagree, so filtered views are rejected. Clear the filter or
``purge_vertices`` / ``purge_edges`` before measuring.
"""

from __future__ import annotations

from graph_tool.all import Graph
from graph_tool.topology import topological_sort
from numpy import array, asarray, flatnonzero, integer


def _filter_property(filtered) -> object | None:
    """Return the property map from a graph-tool filter result, if any."""
    if filtered is None:
        return None
    if isinstance(filtered, tuple):
        return filtered[0]
    return filtered


def assert_no_active_filters(g: Graph) -> None:
    """Raise if *g* has an active vertex or edge filter.

    Parameters
    ----------
    g : Graph
        Graph to inspect.

    Raises
    ------
    ValueError
        If a vertex or edge filter is active.
    """
    vertex_filter = _filter_property(g.get_vertex_filter())
    edge_filter = _filter_property(g.get_edge_filter())
    if vertex_filter is not None or edge_filter is not None:
        raise ValueError(
            "Active graph-tool vertex or edge filters are not supported. "
            "Filtered views keep the full property-map index space, so "
            "per-vertex results would not match the visible subgraph. "
            "Clear the filters or purge the graph first."
        )


def as_vertex_index(g: Graph, root: int) -> int:
    """Return *root* as a Python int after checking it is a vertex index.

    The upper bound is ``num_vertices(ignore_filter=True)`` when that argument
    exists, so a filtered graph can still name a vertex in the full index
    space. Traversals themselves still obey an active filter.

    Parameters
    ----------
    g : Graph
        Graph whose vertex index space is checked.
    root : int
        Candidate vertex index. Booleans are rejected.

    Returns
    -------
    int
        Validated vertex index.

    Raises
    ------
    TypeError
        If *root* is not an integer.
    ValueError
        If *root* is outside the vertex index space.
    """
    if isinstance(root, bool) or not isinstance(root, (int, integer)):
        raise TypeError(f"root must be an integer vertex index, got {type(root).__name__}.")
    index = int(root)
    try:
        n = int(g.num_vertices(ignore_filter=True))
    except TypeError:
        n = int(g.num_vertices())
    if index < 0 or index >= n:
        raise ValueError(f"root index {index} is outside [0, {n}).")
    return index


def validate_rooted_tree(g: Graph) -> int:
    """Return the root index of a directed rooted tree.

    A valid input is directed, non-empty, has exactly one in-degree 0 vertex,
    every other vertex has in-degree 1, and a topological order covers every
    vertex (no directed cycle, no unreachable component). A single vertex is
    valid.

    Parameters
    ----------
    g : Graph
        Candidate morphology graph.

    Returns
    -------
    int
        Index of the unique root (in-degree 0).

    Raises
    ------
    ValueError
        If *g* is undirected, empty, filtered, cyclic, disconnected, or not
        a single arborescence.

    Examples
    --------
    >>> from graph_tool.all import Graph
    >>> g = Graph(directed=True)
    >>> g.add_vertex(1)
    >>> validate_rooted_tree(g)
    0
    """
    assert_no_active_filters(g)
    if not g.is_directed():
        raise ValueError("Graph must be directed.")
    n = int(g.num_vertices())
    if n == 0:
        raise ValueError("Graph is empty; a rooted tree needs exactly one root.")

    # ``.a`` is a view of the property map. Copy it before the temporary map
    # is freed; a bare ``asarray(map.a)`` reads a dangling buffer.
    in_map = g.degree_property_map("in")
    in_degree = array(in_map.a, copy=True)
    if in_degree.shape[0] != n:
        raise ValueError(
            "In-degree property length does not match num_vertices(). "
            "Clear graph filters or purge deleted vertices."
        )
    roots = flatnonzero(in_degree == 0)
    if roots.size != 1:
        raise ValueError(f"Expected exactly one root, found {int(roots.size)}.")
    if (in_degree > 1).any():
        raise ValueError("Every non-root vertex must have exactly one parent.")
    # in-degree 0 is the root; every other vertex must have in-degree 1.
    # Vertices with in-degree > 1 were already rejected. in-degree < 0 cannot
    # occur. The only remaining illegal value would be a second 0, already
    # counted in ``roots``.

    topo = asarray(topological_sort(g))
    if int(topo.size) != n:
        raise ValueError(
            "Topological order did not cover every vertex. "
            "The graph has a directed cycle or is not a single arborescence."
        )
    return int(roots[0])
