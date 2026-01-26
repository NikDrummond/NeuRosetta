# functions to get node indicies from graphs
from typing import List
from functools import partial

from numpy import ndarray, where, unique, concatenate, isin, array
from graph_tool.all import bfs_iterator

from ..core import _Tree, _Forest


### Node indices
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
    return int(where(tree.graph.degree_property_map("in").a == 0)[0][0])


def get_root_forest(
    forest: _Forest,
    *,
    parallel: bool = True,
    max_workers: int | None = 4,
    progress: bool = True,
) -> ndarray:
    """Get all Neuron root node indicies

    Parameters
    ----------
    forest : Forest
        Forest of Neuron trees
    parallel : bool, optional
        If you want to run in parallel, by default True
    max_workers : int | None, optional
        Max number of threads to use, by default 4
    progress : bool, optional
        Show progress bar, by default True

    Returns
    -------
    ndarray
        Array of root node indicies
    """
    return array(
        forest._map(
            get_root, parallel=parallel, max_workers=max_workers, show_progress=progress
        )
    )


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
    return where(tree.graph.degree_property_map("out").a == 0)[0]

def get_leaves_forest(forest: _Forest, *, parallel: bool = True, max_workers: int = 4, progress:bool = True) -> List[ndarray]:
    """Get list of arrays of leaf node indicies

    Parameters
    ----------
    forest : Forest
        Forest of Neuron trees
    parallel : bool, optional
        If you want to run in parallel, by default True
    max_workers : int, optional
        Maximum number of threads, by default 4
    progress : bool, optional
        Show progress bar, by default True

    Returns
    -------
    List[ndarray]
        List of numpy.arrays of leaf node indicies for each neuron tree in forest.
    """
    return forest._map(get_leaves, parallel = parallel, max_workers = max_workers, show_progress = progress)

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
    return where(tree.graph.degree_property_map("out").a > 1)[0]

def get_branches_forest(forest: _Forest, parallel: bool = True, max_workers:int = 4, progress: bool = True) -> List[ndarray]:
    """Get list of arrays of branch node indices

    Parameters
    ----------
    forest : Forest
        Forest of Neuron trees
    parallel : bool, optional
        If you want to run in parallel, by default True
    max_workers : int, optional
        Maximum number of threads, by default 4
    progress : bool, optional
        Show progress bar, by default True

    Returns
    -------
    List[ndarray]
        List of numpy.arrays of branch node indicies for each neuron tree in forest.
    """
    return forest._map(get_branches, parallel = parallel, max_workers = max_workers, show_progress = progress)


def get_core_inds(tree: _Tree, include_root: bool = True) -> ndarray:
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

def get_core_inds_forest(forest: _Forest, include_root:bool = True, parallel: bool = True, max_workers: int = 4, progress: bool = True) -> List[ndarray]:
    """ Get index of core nodes (branch and leaf nodes, optionally including root) for all neurons in forest


    Parameters
    ----------
    forest : _Forest
        Forest of Neuron trees
    include_root : bool, optional
        If true, root is included in the index, by default True
    parallel : bool, optional
        If you want to run in parallel, by default True
    max_workers : int, optional
        maximum number of threads, by default 4
    progress : bool, optional
        If you want to show progress, by default True

    Returns
    -------
    List[ndarray]
        List of arrays of core (root (optional), leaf and branch node) indicies fr all neurons in forest.
    """

    fn = partial(get_core_inds, include_root=include_root)
    return forest._map(fn, parallel=parallel, max_workers=max_workers, show_progress = progress)

def get_edges(
    tree: _Tree, root: int | None = None, subset: str | None = None
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

def get_edges_forest(forest: _Forest, root: int | None = None, subset: str | None = None, parallel: bool = True, max_workers: int = 4, progress:bool = True) -> List[ndarray]:
    """List of arrays for all parent -> child edge node indicies in all neuron trees in forest

    Parameters
    ----------
    forest : Forest
        _description_
    root : int | None, optional
        _description_, by default None
    subset : str | None, optional
        _description_, by default None
    parallel : bool, optional
        _description_, by default True
    max_workers : int, optional
        _description_, by default 4
    progress : bool, optional
        _description_, by default True

    Returns
    -------
    List[ndarray]
        _description_
    """
    fn = partial(get_edges, root=root, subset = subset)
    return forest._map(fn, parallel=parallel, max_workers=max_workers, show_progress = progress)