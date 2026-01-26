# functions to get coordinates from graphs

from numpy import ndarray
from typing import Tuple

from ..core import _Tree
from typing import List
from ..errors.errors import _raise_internal_property
from .vertex_inds import get_edges


def vertex_coordinates(tree: _Tree, subset: int | List | bool = None) -> ndarray:
    """Returns an n by 3 np.array of node coordinates within the neuron

    Parameters
    ----------
    tree : Tree
        Neuron Tree
    subset : int | List | bool, optional
        Subset of node indices, if you only want some of the node coordinates, by default None (which returns all)

    Returns
    -------
    ndarray
        Numpy array of node coordinates
    """

    _raise_internal_property(tree.graph, "coordinates")

    coords = tree.graph.vp["coordinates"].get_2d_array().T

    if subset is not None:
        coords = coords[subset]

    return coords


def edge_coordinates(
    tree: _Tree, root: int | None = None, subset: str | None = None
) -> Tuple[ndarray, ndarray]:
    """Returns tuple of n by 3 np.arrays with coordinates of source and target nodes for each edge (the start and stop point)

    Parameters
    ----------
    tree : Tree
        Neuron tree
    root : int | None, optional
        If provided, will return all edges downstream of the given root in a breadth first search ordering, If not provided, we default to the root node of the neuron. By default None
    subset : str | None, optional
        Can be None (default), 'Internal', or 'External'. Subset is generated after rooting, so if you provide a root the subset will be downstream of this.
        If None, all edges are returned.
        If 'Internal' only internal edges are returned (those with no leaf node as the target)
        if 'External' ony external edges are returned (those terminating in a leaf node)

    Returns
    -------
    Tuple[ndarray,ndarray]
        Tuple of node coordinates paired for each edge, going parent to child.
    """

    edges = get_edges(tree, root, subset)
    coords = vertex_coordinates(tree)
    p1 = coords[edges[:, 0]]
    p2 = coords[edges[:, 1]]
    return p1, p2
