"""Functions for editing trees"""
from graph_tool.all import Graph

from ...core import _Tree
from ...utils.graph_utils import reduce_graph, reroot_graph

def reduce_tree(tree: _Tree, inplace:bool = False) -> Graph | None:
    """remove transitive nodes from graph"""
    g = reduce_graph(tree.graph)

    if inplace:
        tree.graph = g
        tree.metadata = g.gp['metadata']
        return None
    return g

def reroot_tree(tree: _Tree, root: int, inplace: bool = False) -> Graph | None:
    """reroot tree to new root"""
    g = reroot_graph(tree.graph, root)

    if inplace:
        tree.graph = g
        return None
    return g
