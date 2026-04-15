"""Functions for grabbing vertex coordinates"""

from typing import List, Tuple
from numpy import ndarray
from graph_tool.all import Graph

from .gt_properties import raise_internal_property_missing
from .vertex_inds import edge_indices, subtree_indices


def vertex_coordinates(
    g: Graph, subset: int | List | None = None, SoA: bool = False
) -> ndarray:
    """Get coordinates of vertices

    Parameters
    ----------
    g : Graph
        Graph object
    subset : int | List | None, optional
        Subset of ndoe indicies if only a subset of coordinates is wanted, by default None
    SoA : bool, optional
        Whether to return in Sequence of Arrays format (Dimensions by vertices). 
        If false return vertices by dimensions, by default False

    Returns
    -------
    ndarray
        Array of vertex coordinates
    """

    # Make sure we have coordinates
    raise_internal_property_missing(g, "coordinates", "v")

    # get coordinates
    coords = g.vp["coordinates"].get_2d_array()

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
    """Get coordinates of vertices for subtree defined by a given root vertex

    Parameters
    ----------
    g : Graph
        Graph object
    root : int
        root vertex index
    traversal_order : str, optional
        Specifies traversal order, must be Breadth or Depth, by default "Breadth"
    SoA : bool, optional
        Whether to return in Sequence of Arrays format (Dimensions by vertices). 
        If false return vertices by dimensions, by default False

    Returns
    -------
    ndarray
        Array of vertex coordinates in subtree
    """

    # Make sure we have coordinates
    raise_internal_property_missing(g, "coordinates", "v")
    # get subtree vert. inds
    sub_inds = subtree_indices(g, root, traversal_order)
    # get coordinates with subset
    return vertex_coordinates(g, subset=sub_inds, SoA=SoA)


def edge_coordinates(g: Graph, SoA: bool = False) -> Tuple[ndarray, ndarray]:
    """Get source - target coordinates for edges.

    Parameters
    ----------
    g : Graph
        graph object
    SoA : bool, optional
        Whether to return in Sequence of Arrays format (Dimensions by vertices). 
        If false return vertices by dimensions, by default False.

    Returns
    -------
    Tuple[ndarray,ndarray]
        Source and Target coordinate arrays over edges.
    """

    # Make sure we have coordinates
    raise_internal_property_missing(g, "coordinates", "v")

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
    """get Source Target coordinates for a subtree specified by root vertex.

    Parameters
    ----------
    g : Graph
        Graph object
    root : int
        root vertex index
    traversal_order : str, optional
        Specifies traversal order, must be Breadth or Depth, by default "Breadth"
    SoA : bool, optional
        Whether to return in Sequence of Arrays format (Dimensions by vertices). 
        If false return vertices by dimensions, by default False

    Returns
    -------
    Tuple[ndarray, ndarray]
        Source and Target coordinate arrays over edges in subtree defined by root.
    """

    # Make sure we have coordinates
    raise_internal_property_missing(g, "coordinates", "v")

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
