# functions to get coordinates from graphs

from numpy import ndarray
from typing import Tuple

from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property
from .vertex_inds import get_edges

def vertex_coordinates(tree:_Tree, subset : int | List | bool = None) -> ndarray:
    """Returns an n by 3 np.array of node coordinates within the neuron 

    Parameters
    ----------
    tree : Tree_graph
        Neuron Tree
    subset : int | List | bool, optional
        Subset of node indices, if you only want some of the node coordinates, by default None (which returns all)

    Returns
    -------
    ndarray
        Numpy array of node coordinates
    """

    _check_internal_property(tree.graph,'coordinates')

    coords = tree.graph.vp['coordinates'].get_2d_array().T

    if subset is not None:
        coords = coords[subset]

    return coords

def edge_coordinates(tree:_Tree, root:int|None = None,subset:str|None = None) -> Tuple[ndarray,ndarray]:
    """

    Parameters
    ----------
    tree : Tree_graph
        Neuron_tree
    root : int | None, optional
        _description_, by default None
    subset : str | None, optional
        _description_, by default None

    Returns
    -------
    Tuple[ndarray,ndarray]
        _description_
    """

    edges = get_edges(tree, root, subset)
    coords = vertex_coordinates(tree)
    p1 = coords[edges[:,0]]
    p2 = coords[edges[:,1]]
    return p1, p2