"""Functions for tree structural metrics."""

from numpy import ndarray, bincount, median

from ...core import _Tree
from ...utils.graph_utils import (
    count_core_vertices,
    count_branches,
    root_index,
    TreeDepthVisitor,
    g_has_property,
    get_property,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_traversals import breadth_first_search


# depths
def get_node_depth(
    tree: _Tree, root: int | None = None, bind: bool = True
) -> ndarray | None:
    """Compute node depths from root using breadth-first search.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        Root vertex index. If None, uses the tree's root index. By default None.
    bind : bool, optional
        If True, binds the "depth" vertex property to the graph and returns None.
        If False, returns the depth array. By default True.

    Returns
    -------
    ndarray | None
        Array of node depths if bind=False, otherwise None.
    """
    if root is None:
        root = root_index(tree.graph)

    out = breadth_first_search(
        tree=tree,
        visitor=TreeDepthVisitor,
        init_vertex_properties={"depth": "int"},
        root=root,
        bind=bind,
    )

    return out["depth"].a if not bind else None


# lazy helper
def _fetch_depth(tree):
    if g_has_property(tree.graph, "depth", "v"):
        return get_property(tree.graph, "depth", "v")
    return get_node_depth(tree, bind=False)


def get_max_depth(tree: _Tree) -> int:
    """Get the maximum node depth from the root.

    Depth is the number of edges on the shortest path from the root to a node.
    Uses the bound ``depth`` vertex property when present, otherwise computes
    depths via :func:`get_node_depth`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Maximum node depth in the tree.
    """
    return int(_fetch_depth(tree).max())


def get_mean_depth(tree: _Tree) -> float:
    """Get the mean node depth from the root.

    Uses the bound ``depth`` vertex property when present, otherwise computes
    depths via :func:`get_node_depth`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Mean depth over all nodes.
    """
    return float(_fetch_depth(tree).mean())


def get_median_depth(tree: _Tree) -> float:
    """Get the median node depth from the root.

    Uses the bound ``depth`` vertex property when present, otherwise computes
    depths via :func:`get_node_depth`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Median depth over all nodes.
    """
    return float(median(_fetch_depth(tree)))


# width (at each depth)
def get_tree_widths(tree: _Tree) -> ndarray:
    """Get the number of nodes at each depth level.

    Returns an array where index *d* is the count of nodes at depth *d*.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    ndarray
        Node counts per depth level (length ``max_depth + 1``).
    """
    return bincount(_fetch_depth(tree))


def get_max_width(tree: _Tree) -> int:
    """Get the maximum width (node count) at any depth level.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Maximum number of nodes at a single depth.
    """
    return int(get_tree_widths(tree).max())


def get_mean_width(tree: _Tree) -> float:
    """Get the mean width across depth levels.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Mean number of nodes per depth level.
    """
    return float(get_tree_widths(tree).mean())


def get_median_width(tree: _Tree) -> float:
    """Get the median width across depth levels.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Median number of nodes per depth level.
    """
    return float(median(get_tree_widths(tree)))


### binary ratio
def get_binary_ratio(tree: _Tree) -> float:
    """Compute the binary branching ratio of the tree.

    Defined as ``count_branches / ((count_core_nodes - 1) / 2)``. A value of
    1.0 indicates a fully binary tree; lower values indicate fewer branch
    points relative to a binary topology.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Binary branching ratio.
    """
    return count_branches(tree.graph) / ((count_core_vertices(tree.graph) - 1) / 2)


enrich_tree_graph_docstrings(globals())
