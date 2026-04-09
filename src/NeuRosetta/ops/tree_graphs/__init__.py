"""Tree graph operations."""

from .vertex_inds import get_root, get_leaves, get_branches, get_core_inds, get_edges
from .coordinates import vertex_coordinates, edge_coordinates
from .counting import (
    count_roots,
    count_vertices,
    count_edges,
    count_branches,
    count_leaves,
    count_transitive_nodes,
)
from .tree_checks import is_Reduced, update_reduced, has_property
from .degrees import get_degrees, degree_distribution
from .path_lengths import euclidean_edge_length
from .traversals import BF_search, DF_search, compute_depths, compute_post_order, reduce_graph

__all__ = [
    "get_root",
    "get_leaves",
    "get_branches",
    "get_core_inds",
    "get_edges",
    "vertex_coordinates",
    "edge_coordinates",
    "count_roots",
    "count_vertices",
    "count_edges",
    "count_branches",
    "count_leaves",
    "count_transitive_nodes",
    "is_Reduced",
    "update_reduced",
    "has_property",
    "get_degrees",
    "degree_distribution",
    "euclidean_edge_length",
    "BF_search",
    "DF_search",
    "compute_depths",
    "compute_post_order",
    "reduce_graph",
]
