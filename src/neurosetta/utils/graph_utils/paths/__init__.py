"""Path retrieval, distances, ancestry, and path geometry.

Low-level functions take a ``graph_tool.Graph``. They do not bind results
onto the graph. Tree and Forest wrappers live in ``ops/tree_graphs``.

Directed paths follow parent-to-child edges. ``directed=False`` uses the
unique undirected path and leaves the stored orientation in place.

``Path_length`` is physical cable, including collapsed section length on a
reduced tree. ``Euclidean_length`` is the straight edge. Endpoint
displacement is a third quantity: the straight line between two vertex
coordinates, independent of either edge property.
"""

from .ancestry import (
    ancestor_indices,
    branch_path,
    is_ancestor,
    lowest_common_ancestor,
    parent_index,
)
from .distances import (
    pairwise_path_lengths,
    path_cumulative_length,
    path_edge_count,
    path_length,
    path_lengths_to_vertices,
    root_distance,
    source_distances,
)
from .geometry import path_coordinates, path_euclidean_distance, path_tortuosity
from .morphology import path_branch_count, path_section_count
from .profiles import path_edge_values, path_vertex_values
from .retrieval import path_edges, path_exists, path_vertices

__all__ = [
    "ancestor_indices",
    "branch_path",
    "is_ancestor",
    "lowest_common_ancestor",
    "pairwise_path_lengths",
    "parent_index",
    "path_branch_count",
    "path_coordinates",
    "path_cumulative_length",
    "path_edge_count",
    "path_edge_values",
    "path_edges",
    "path_euclidean_distance",
    "path_exists",
    "path_length",
    "path_lengths_to_vertices",
    "path_section_count",
    "path_tortuosity",
    "path_vertex_values",
    "path_vertices",
    "root_distance",
    "source_distances",
]
