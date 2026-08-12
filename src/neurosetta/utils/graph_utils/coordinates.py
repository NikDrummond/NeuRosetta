"""Functions for extracting vertex and edge coordinates from graphs."""
from __future__ import annotations

from typing import List, Tuple
from numpy import ndarray, stack
from graph_tool.all import Graph

from .gt_properties import raise_internal_property_missing
from .vertex_inds import edge_indices, subtree_indices


def vertex_coordinates(
    g: Graph, subset: int | List[int] | None = None, SoA: bool = False
) -> ndarray:
    """Get coordinates of vertices.

    Parameters
    ----------
    g : Graph
        Graph object with vertex property 'coordinates'.
    subset : int | List[int] | None, optional
        Subset of node indices if only a subset of coordinates is wanted.
        If int, treated as a single index. If list, treated as array of indices.
        By default None (all vertices).
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions (vertices x 3), by default False.

    Returns
    -------
    ndarray
        Array of vertex coordinates with shape (3, N) if SoA=True,
        or (N, 3) if SoA=False, where N is the number of vertices.
    """
    # Make sure we have x,y,z
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")

    # get coordinates
    coords = stack([g.vp[p].a for p in ["x","y","z"]], axis = 0)

    # subset if needed
    if subset is not None:
        coords = coords[:, subset]

    # sort SoA
    if not SoA:
        coords = coords.T

    return coords


def vertex_coordinates_subtree(
    g: Graph, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> ndarray:
    """Get coordinates of vertices for subtree defined by a given root vertex.

    Parameters
    ----------
    g : Graph
        Graph object with vertex property 'coordinates'.
    root : int
        Root vertex index defining the subtree.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth",
        by default "Breadth".
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions, by default False.

    Returns
    -------
    ndarray
        Array of vertex coordinates in subtree with shape (3, N) if SoA=True,
        or (N, 3) if SoA=False, where N is the number of vertices in the subtree.
    """
    # Make sure we have coordinates
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")
    # get subtree vert. inds
    sub_inds = subtree_indices(g, root, traversal_order)
    # get coordinates with subset
    return vertex_coordinates(g, subset=sub_inds, SoA=SoA)


def edge_coordinates(g: Graph, SoA: bool = False) -> Tuple[ndarray, ndarray]:
    """Get source and target coordinates for all edges.

    Parameters
    ----------
    g : Graph
        Graph object with vertex property 'coordinates'.
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions, by default False.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Source and target coordinate arrays over edges, each with shape (3, E)
        if SoA=True, or (E, 3) if SoA=False, where E is the number of edges.
    """
    # Make sure we have coordinates
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")

    # get edges
    edges = edge_indices(g)
    # get coordinates
    coords = vertex_coordinates(g)
    p1 = coords[edges[:, 0]]
    p2 = coords[edges[:, 1]]
    # sort SoA
    if SoA:
        p1, p2 = p1.T, p2.T

    return p1, p2


def edge_coordinates_subtree(
    g: Graph, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> Tuple[ndarray, ndarray]:
    """Get source and target coordinates for edges in a subtree.

    Parameters
    ----------
    g : Graph
        Graph object with vertex property 'coordinates'.
    root : int
        Root vertex index defining the subtree.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth",
        by default "Breadth".
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions, by default False.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Source and target coordinate arrays over edges in subtree, each with
        shape (3, E) if SoA=True, or (E, 3) if SoA=False, where E is the
        number of edges in the subtree.
    """
    # Make sure we have coordinates
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")

    # get coordinates
    coords = vertex_coordinates(g)
    # get subtree edge inds
    edges = edge_indices(g, root, traversal_order)

    p1 = coords[edges[:, 0]]
    p2 = coords[edges[:, 1]]
    # sort SoA
    if SoA:
        p1, p2 = p1.T, p2.T

    return p1, p2