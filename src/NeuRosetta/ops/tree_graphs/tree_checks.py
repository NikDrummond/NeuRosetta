""" various checks for tree graphs """

from ...core import _Tree
from ...utils.graph_utils import g_has_property, count_transitive_vertices


def check_reduced(tree: _Tree):
    """Check if the given tree has no nodes with in-deg == out_deg == 1"""
    return count_transitive_vertices(tree.graph) == 0

def update_reduced(tree: _Tree):
    """Update isReduced in metadata"""
    if check_reduced(tree):
        tree.graph.gp['metadata']['isReduced'] = True
    else:
        tree.graph.gp['metadata']['isReduced'] = False

def tree_has_property(tree: _Tree, prop: str, level: str = "all"):
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
