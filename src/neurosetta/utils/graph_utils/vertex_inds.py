"""Graph functions to get indices of vertices."""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import array, concatenate, ndarray, unique, where

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


def edge_indices(g: Graph, root: int | None = None, traversal_order: str = "Breadth") -> ndarray:
    """Return edge index array.

    Parameters
    ----------
    g : Graph
        Directed tree graph.
    root : int or None, optional
        Root vertex index for subtree extraction. If None, return all edges.
        By default None.
    traversal_order : {"Breadth", "Depth"}, optional
        Traversal order for subtree extraction. Only used if root is provided.
        By default "Breadth".

    Returns
    -------
    ndarray
        Array of edges with shape (n_edges, 2), where each row is
        (source, target).

    Raises
    ------
    ValueError
        If traversal_order is not "Breadth" or "Depth".
    """
    if root is None:
        edges = g.get_edges()
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
