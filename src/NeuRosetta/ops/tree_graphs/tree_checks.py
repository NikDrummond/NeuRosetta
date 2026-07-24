"""Various checks for tree graphs."""

from ...core import _Tree
from ...utils.graph_utils import g_has_property, count_transitive_vertices


def check_reduced(tree: _Tree) -> bool:
    """Check if the given tree has no transitive vertices (in-degree == out-degree == 1).

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    bool
        True if the tree has no transitive vertices (is reduced), False otherwise.
    """
    return count_transitive_vertices(tree.graph) == 0


def update_reduced(tree: _Tree) -> None:
    """Update the 'isReduced' flag in the tree's metadata.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    """
    if check_reduced(tree):
        tree.graph.gp["metadata"]["isReduced"] = True
    else:
        tree.graph.gp["metadata"]["isReduced"] = False


def has_property(tree: _Tree, prop: str, level: str = "all") -> bool:
    """Check if the tree has a specific internal property.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    prop : str
        String name of internal property map.
    level : str, optional
        Level to check: "all" (default), "g" (graph), "v" (vertex), or "e" (edge).

    Returns
    -------
    bool
        True if graph has property.

    Raises
    ------
    AttributeError
        If level is not None or one of ['g', 'v', 'e'].
    """
    return g_has_property(tree.graph, prop, level)