# functions to get node indicies from graphs

from numpy import ndarray, where, unique, concatenate

from ..core import _Tree
from ..errors.errors import _check_internal_property

### Node indices


def get_root(tree: _Tree) -> int:
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
    return where(tree.graph.degree_property_map("in").a == 0)[0][0]


def get_leaves(tree: _Tree) -> ndarray:
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


def get_branches(tree: _Tree) -> ndarray:
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


def get_core_inds(tree: _Tree, include_root: bool = True) -> ndarray:
    """
    Get index of core nodes (branch and leaf nodes, optionally including root)
    Parameters
    ----------
    tree : Tree_graph
        Neuron tree
    include_root : bool, optional
        If True, includes root node index
    Returns
    -------
    np.ndarray
        Index of core nodes
    """

    l_inds = get_leaves(tree)
    b_inds = get_branches(tree)

    inds = unique(concatenate([l_inds, b_inds]))

    root = get_root()

    if ~include_root:
        inds = inds[inds != get_root(tree)]

    return inds

