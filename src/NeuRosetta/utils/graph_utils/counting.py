""" functions for counting nodes of different types"""
from numpy import where #, ndarray, array
from graph_tool.all import Graph

from .vertex_inds import leaf_indices, branch_indices

def count_roots(g: Graph) -> int:
    """Count the number of root nodes"""
    return len(where(g.degree_property_map("in").a == 0))

def count_vertices(g: Graph) -> int:
    """Count the number of vertices in a graph"""
    return g.num_vertices()

def count_edges(g: Graph) -> int:
    """Count the number of edges in a graph"""
    return g.num_edges()

def count_leaves(g: Graph) -> int:
    """Count the number of leaf vertices in a graph"""
    return len(leaf_indices(g))

def count_branches(g: Graph) -> int:
    """Count the number of branch vertices in a graph"""
    return len(branch_indices(g))

def count_transitive_vertices(g: Graph) -> int:
    """Count the nnumber of transitive (in degree == out degree == 1) vertices in a graph"""
    return int(
        sum(
            (g.degree_property_map("out").a == 1)
            & (g.degree_property_map("in").a == 1)
        )
    )
