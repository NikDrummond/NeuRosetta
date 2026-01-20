### boring functions for counting things
from numpy import where

from ..core import _Tree

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

def count_vertices(tree: )