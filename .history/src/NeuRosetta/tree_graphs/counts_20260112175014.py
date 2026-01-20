### boring functions for counting things
from numpy import where

from ..core import _Tree
from .

def count_roots(tree: _Tree) -> int:
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
    return len(where(tree.graph.degree_property_map("in").a == 0))

def count_vertices(tree: _Tree) -> int:
    """

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """
    return tree.graph.num_vertices()

def count_edges(tree: _Tree) -> int:
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
    return 