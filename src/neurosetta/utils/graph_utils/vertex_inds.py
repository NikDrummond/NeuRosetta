"""Graph functions to get indices of vertices."""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import argsort, array, concatenate, empty, full, ndarray, unique, where

from .traversals import bf_iterator, df_iterator
from .tree_validation import as_vertex_index


def root_index(g: Graph) -> int:
    """Return index of root node (in-degree == 0).

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    int
        Vertex index of the root node.
    """
    return int(where(g.degree_property_map("in").a == 0)[0][0])


def leaf_indices(g: Graph) -> ndarray:
    """Return array of indices of leaves (out-degree == 0).

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    ndarray
        Array of leaf vertex indices.
    """
    return where(g.degree_property_map("out").a == 0)[0]


def branch_indices(g: Graph) -> ndarray:
    """Return array of indices of branch points (out-degree > 1).

    Parameters
    ----------
    g : Graph
        Directed tree graph.

    Returns
    -------
    ndarray
        Array of branch vertex indices.
    """
    return where(g.degree_property_map("out").a > 1)[0]


def core_indices(g: Graph, include_root: bool = True) -> ndarray:
    """Return array of indices of core vertices (branches and leaves).

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    include_root : bool, optional
        If True, include the root vertex in the result. If False, exclude it.
        By default True.

    Returns
    -------
    ndarray
        Array of core vertex indices.
    """
    l_inds = leaf_indices(g)
    b_inds = branch_indices(g)

    inds = unique(concatenate([l_inds, b_inds]))

    root = root_index(g)

    root_in = root in inds

    if include_root and not root_in:
        inds = concatenate(([root], inds))
    elif not include_root and root_in:
        inds = inds[inds != root]
    return inds


def _edges_in_index_order(g: Graph) -> ndarray:
    """Return live ``(source, target)`` rows sorted by ``g.edge_index``.

    ``Graph.get_edges()`` is not edge-index order for graphs built with
    ``hashed=True`` (SWC import, reduction, rerooting). When edge indices
    are exactly ``0 .. num_edges()-1``, row ``i`` is edge index ``i`` and
    lines up with ``ep.*.a``.

    ``remove_edge`` without ``purge_edges`` leaves gaps, so a surviving
    edge can have an index greater than ``num_edges()-1``. Those graphs
    still get one row per live edge, in increasing edge-index order. Row
    ``i`` is then not edge index ``i``. Callers that stored an edge index
    must not use it as a row until the graph is purged.
    """
    n = int(g.num_edges())
    out = empty((n, 2), dtype=int)
    if n == 0:
        return out
    indices = empty(n, dtype=int)
    for i, edge in enumerate(g.edges()):
        indices[i] = int(g.edge_index[edge])
        out[i, 0] = int(edge.source())
        out[i, 1] = int(edge.target())
    order = argsort(indices, kind="stable")
    return out[order]


def source_index_by_edge_index(g: Graph) -> ndarray:
    """Source vertex of each live edge, addressable by ``g.edge_index``.

    The array length is ``max(edge_index) + 1``. Indices of edges deleted
    but not purged are ``-1``.
    """
    n = int(g.num_edges())
    if n == 0:
        return empty(0, dtype=int)
    max_index = -1
    pairs: list[tuple[int, int]] = []
    for edge in g.edges():
        index = int(g.edge_index[edge])
        pairs.append((index, int(edge.source())))
        if index > max_index:
            max_index = index
    sources = full(max_index + 1, -1, dtype=int)
    for index, source in pairs:
        sources[index] = source
    return sources


def edge_indices(g: Graph, root: int | None = None, traversal_order: str = "Breadth") -> ndarray:
    """Return edge index array.

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    root : int or None, optional
        Root vertex index for subtree extraction. If None, return all edges
        in graph-tool edge-index order (the order of ``ep.*.a``).
        By default None.
    traversal_order : {"Breadth", "Depth"}, optional
        Traversal order for subtree extraction. Only used if root is provided.
        By default "Breadth".

    Returns
    -------
    ndarray
        Array of edges with shape (n_edges, 2), where each row is
        (source, target). With ``root=None`` and no unpurged deletions,
        row ``i`` is the edge whose ``g.edge_index`` is ``i``. After
        ``remove_edge`` without ``purge_edges``, rows are the surviving
        edges in increasing edge-index order.

    Raises
    ------
    ValueError
        If traversal_order is not "Breadth" or "Depth".
    """
    if root is None:
        edges = _edges_in_index_order(g)
    else:
        if traversal_order == "Breadth":
            edges = bf_iterator(g, root, array=True)
        elif traversal_order == "Depth":
            edges = df_iterator(g, root, array=True)
        else:
            raise ValueError(f"traversal_order must be Breadth or Depth, not {traversal_order}")
    return edges


def subtree_indices(g: Graph, root: int, traversal_order: str = "Breadth") -> ndarray:
    """Return vertex indices of subtree rooted at specified vertex.

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    root : int
        Root vertex index of the subtree.
    traversal_order : {"Breadth", "Depth"}, optional
        Traversal order for subtree extraction. By default "Breadth".

    Returns
    -------
    ndarray
        Sorted array of vertex indices in the subtree, including *root*.
        A leaf-rooted subtree is ``array([root])``.

    Raises
    ------
    TypeError
        If *root* is not an integer vertex index.
    ValueError
        If *root* is outside the vertex index space, or *traversal_order*
        is not ``"Breadth"`` or ``"Depth"``.

    Notes
    -----
    Ordering is the sorted unique vertex set, for both traversal orders.
    Breadth-first and depth-first therefore return the same indices.
    """
    root = as_vertex_index(g, root)
    if traversal_order == "Breadth":
        edges = bf_iterator(g, root, array=True)
    elif traversal_order == "Depth":
        edges = df_iterator(g, root, array=True)
    else:
        raise ValueError(f"traversal_order must be Breadth or Depth, not {traversal_order}")
    if edges is None or array(edges).size == 0:
        return array([root], dtype=int)
    verts = unique(array(edges))
    if root not in verts:
        verts = unique(concatenate((verts, [root])))
    return verts


def bifurcation_indices(g: Graph, include_root: bool = False) -> ndarray:
    """Return indices of bifurcations in a tree graph.

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    include_root : bool, optional
        If True, include the root index when it is a bifurcation. By default False.

    Returns
    -------
    ndarray
        Array of bifurcation vertex indices.
    """
    if include_root:
        return where(g.degree_property_map("out").a == 2)[0]
    return where((g.degree_property_map("out").a == 2) & (g.degree_property_map("in").a != 0))[0]
