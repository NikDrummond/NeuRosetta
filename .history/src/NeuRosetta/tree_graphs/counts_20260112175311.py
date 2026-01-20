### boring functions for counting things
from numpy import where

from ..core import _Tree
from .vertex_inds import get_leaves, get_branches

def count_roots(tree: _Tree) -> int:
    """
    Returns the number of root nodes (those with in degree = 0)

    Parameters
    ----------
    tree : Tree_graph
        Neuron tree

    Returns
    -------
    int
        Number of root nodes
    """
    return len(where(tree.graph.degree_property_map("in").a == 0))

def count_vertices(tree: _Tree) -> int:
    """
    returns the number of nodes in the tree

    Parameters
    ----------
    tree : Tree_graph
        Neuron tree

    Returns
    -------
    int
        Number of nodes in the tree graph
    """
    return tree.graph.num_vertices()

def count_edges(tree: _Tree) -> int:
    """
    Returns the number of edges in the tree graph

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """
    return tree.graph.num_edges()

def count_leaves(tree: _Tree) -> int:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """
    return len(get_leaves(tree))

def count_branches(tree: _Tree) -> int:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """
    return len(get_branches(tree))