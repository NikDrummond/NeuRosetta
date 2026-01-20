from numpy import ndarray, where

from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property

### Node indicies

def get_root(tree:_Tree) -> int:
    """
    Get index of root node
    Parameters
    ----------
    tree : Tree_graph
        Neuron tree

    Returns
    -------
    int
        Index of root node
    """
    return where(tree.graph.degree_property_map("in").a == 0)[0]

def get_leaves(tree:_Tree) -> ndarray:
    """
    Get index of leaves nodes
    Parameters
    ----------
    tree : Tree_graph
        Neuron tree

    Returns
    -------
    np.ndarray
        Index of leaves nodes
    """
    return where(tree.graph.degree_property_map("out").a == 0)[0]

def get_branches(tree:_Tree) -> ndarray:
    """
    Get index of branches nodes
    Parameters
    ----------
    tree : Tree_graph
        Neuron tree

    Returns
    -------
    np.ndarray
        Index of branches nodes
    """
    return where(tree.graph.degree_property_map("out").a > 1)[0]

def get_core_inds(tree:_Tree, root = None) -> ndarray:
    """
    Get index of core nodes (branch and leaf nodes, optionally including root)
    Parameters
    ----------
    tree : Tree_graph
        Neuron tree
    root : int, optional
        If True, includes root node index
    Returns
    -------
    np.ndarray
        Index of core nodes
    """

    l_inds = get_leaves(tree)


    
    return where(tree.graph.degree_property_map("out").a > 1)[0]


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