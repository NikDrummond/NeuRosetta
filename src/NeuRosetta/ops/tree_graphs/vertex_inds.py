""" functions to get node indicies from graphs """
from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    root_index,
    leaf_indices,
    branch_indices,
    core_indices,
    edge_indices,
    subtree_indices,
)

def get_root(tree: _Tree) -> int:
    """
    Get index of root node
    Parameters
    ----------
    tree : Tree
        Neuron tree

    Returns
    -------
    int
        Index of root node
    """
    return root_index(tree.graph)


def get_leaves(tree: _Tree) -> ndarray:
    """
    Get index of leaves nodes
    Parameters
    ----------
    tree : Tree
        Neuron tree

    Returns
    -------
    np.ndarray
        Index of leaves nodes
    """
    return leaf_indices(tree.graph)


def get_branches(tree: _Tree) -> ndarray:
    """
    Get index of branches nodes
    Parameters
    ----------
    tree : Tree
        Neuron tree

    Returns
    -------
    np.ndarray
        Index of branches nodes
    """
    return branch_indices(tree.graph)


def get_core_indices(tree: _Tree, include_root: bool = True) -> ndarray:
    """
    Get index of core nodes (branch and leaf nodes, optionally including root)
    Parameters
    ----------
    tree : Tree
        Neuron tree
    include_root : bool, optional
        If True, includes root node index
    Returns
    -------
    np.ndarray
        Index of core nodes
    """

    return core_indices(tree.graph, include_root)

def get_subtree_nodes(tree: _Tree, root: int , traversal_order: str = "Breadth") -> ndarray:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    root : int
        _description_
    traversal_order : str, optional
        _description_, by default "Breadth"

    Returns
    -------
    ndarray
        _description_
    """
    return subtree_indices(tree.graph, root, traversal_order)

def get_edges(
    tree: _Tree, root: int | None = None, traversal_order: str = "Breadth"
) -> ndarray:
    """
    Returns nx2 array of edge indices going parent -> child.

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
    np.ndarray
        nx2 array of node index pairs for each edge.

    Raises
    ------
    ValueError
        If a subset is passed which is not 'Internal' or 'External' or None.
    """

    return edge_indices(tree.graph, root, traversal_order)
