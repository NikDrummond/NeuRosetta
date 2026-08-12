"""Functions for editing trees."""

from __future__ import annotations

from graph_tool.all import Graph

from ...core import _Tree
from ...utils.graph_utils import reduce_graph, reroot_graph


def reduce_tree(tree: _Tree, inplace: bool = False) -> Graph | None:
    """Remove transitive nodes from the tree graph.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    inplace : bool, optional
        If True, replace the tree's graph with the reduced graph in place and
        update metadata. If False, return the reduced graph. By default False.

    Returns
    -------
    Graph | None
        Reduced graph if inplace=False, otherwise None.
    """
    g = reduce_graph(tree.graph)

    if inplace:
        tree.graph = g
        return None
    return g


def reroot_tree(tree: _Tree, root: int, inplace: bool = False) -> Graph | None:
    """Reroot the tree to a new root vertex.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Vertex index to use as the new root.
    inplace : bool, optional
        If True, replace the tree's graph with the rerooted graph in place.
        If False, return the rerooted graph. By default False.

    Returns
    -------
    Graph | None
        Rerooted graph if inplace=False, otherwise None.
    """
    g = reroot_graph(tree.graph, root)

    if inplace:
        tree.graph = g
        return None
    return g
