from numpy import ndarray, where

from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property

### Node indicies

def get_root(tree:_Tree) -> int:
    """
    get index of root node
    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """
    return where(_Tree.graph.degree_property_map("in").a == 0)[0]

### coordinates
def vertex_coordinates(tree:_Tree, subset : int | List | bool = None) -> ndarray:
    """Returns an n by 3 np.array of node coordinates within the neuron 

    Parameters
    ----------
    tree : Tree_graph
        Neuron representations
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