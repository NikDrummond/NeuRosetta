"""Functions to get node indices from graphs."""

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
    """Get index of root node.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Index of root node.
    """
    return root_index(tree.graph)


def get_leaves(tree: _Tree) -> ndarray:
    """Get indices of leaf nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    np.ndarray
        Indices of leaf nodes (out-degree == 0).
    """
    return leaf_indices(tree.graph)


def get_branches(tree: _Tree) -> ndarray:
    """Get indices of branch nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    np.ndarray
        Indices of branch nodes (out-degree > 1).
    """
    return branch_indices(tree.graph)


def get_core_indices(tree: _Tree, include_root: bool = True) -> ndarray:
    """Get indices of core nodes (branch and leaf nodes, optionally including root).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    include_root : bool, optional
        If True, includes root node index. By default True.

    Returns
    -------
    np.ndarray
        Indices of core nodes.
    """
    return core_indices(tree.graph, include_root)


def get_subtree_nodes(tree: _Tree, root: int, traversal_order: str = "Breadth") -> ndarray:
    """Get node indices of subtree rooted at specified vertex.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Root vertex index of the subtree.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth".
        By default "Breadth".

    Returns
    -------
    ndarray
        Sorted array of vertex indices in the subtree.
    """
    return subtree_indices(tree.graph, root, traversal_order)


def get_edges(
    tree: _Tree, root: int | None = None, traversal_order: str = "Breadth"
) -> ndarray:
    """Return n x 2 array of edge indices going parent -> child.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        If provided, returns all edges downstream of the given root in a
        breadth-first search ordering. If None, defaults to the root node
        of the neuron. By default None.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth".
        By default "Breadth".

    Returns
    -------
    np.ndarray
        n x 2 array of node index pairs for each edge.

    Raises
    ------
    ValueError
        If traversal_order is not "Breadth" or "Depth".
    """
    return edge_indices(tree.graph, root, traversal_order)