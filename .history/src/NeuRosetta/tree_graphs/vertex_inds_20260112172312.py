# functions to get node indicies from graphs

from numpy import ndarray, where, unique, concatenate, isin
from graph_tool.all import bfs_iterator

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

    root = get_root(tree)

    # check if the root is in inds
    root_in = root in inds

    # include it or exclude it as we want
    if include_root and not root_in:
        inds = concatenate(([root], inds))
    elif not include_root and root_in:
        inds = inds[inds != root]

    return inds


def get_edges(
    tree: _Tree, root: int | None = None, subset: str | None = None
) -> ndarray:
    """
    Returns nx2 array of edge indicies going parent -> child.

    Parameters
    ----------
    tree : Tree_graph
        Neuron tree
    root : int | None, optional
        If provided, will return all edges downstream of the given root in a breadth first search ordering, If not provided, we default to the root node of the neuron. By default None
    subset : str | None, optional
        Can be None (default), 'Internal', or 'External'
        If None, all edges are returned.
        If 'Internal' only internal edges are returned (those with no leaf node as the target)
        if 'External'

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    ValueError
        _description_
    """

    # get edges
    if root is None:
        edges = tree.graph.get_edges()
    else:
        edges = bfs_iterator(tree.graph, root, array=True)

    ### Subset if needed
    expected_subsets = ["None", "Internal", "External"]
    if subset == None:
        return edges
    elif subset == "Internal":
        l_inds = get_leaves(tree)
        return edges[~isin(edges[:, 1], l_inds)]
    elif subset == "External":
        l_inds = get_leaves(tree)
        return edges[isin(edges[:, 1], l_inds)]
    else:
        raise ValueError(
            f"Given Subset {subset} is not valid, expected one of {expected_subsets}"
        )
