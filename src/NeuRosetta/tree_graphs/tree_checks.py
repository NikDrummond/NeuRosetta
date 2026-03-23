### various checks for tree graphs
from numpy import ndarray, array
from functools import partial

from ..core import _Tree, _Forest
from .counting import count_transitive_nodes
from ..graphs.properties import g_has_property


def is_Reduced(tree: _Tree):
    """Check if the given graph, g, has no nodes with"""
    return count_transitive_nodes(tree) == 0

def update_reduced(tree):
    """Update isReduced in metadata"""
    if is_Reduced(tree):
        tree.graph.gp['metadata']['isReduced'] = True
    else:
        tree.graph.gp['metadata']['isReduced'] = False

def has_property(tree: _Tree, prop: str, level: str | None = None):
    """_summary_

    Parameters
    ----------
    g : Graph
        graph
    prop : str
        string name of internal property map
    level : str | None, optional
        If none, all graph vertex and edge properties are checked (default).
        If 'g', only check graph level internal properties.
        If 'v', only check vertex level internal properties.
        If 'e', only check edge level internal properties.

    Returns
    -------
    bool
        True if graph has property

    Raises
    ------
    AttibuteError
        if level not None or one of ['g', 'v', 'e']
    """

    return g_has_property(tree.graph, prop, level)
