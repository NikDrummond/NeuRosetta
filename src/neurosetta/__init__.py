"""
NeuRosetta: A Python package for morphological analysis and connectomics.

This package provides tools for working with tree-like structures,
spatial coordinates, and graph-based morphological data.
"""

import os

os.environ["OMP_WAIT_POLICY"] = "passive"

from .analysis import reconstruct_neuropil_surface
from .api import Forest, Forest_mesh, Neuropil, Neuropils, Tree, Tree_mesh
from .gui import start_GUI
from .io import export_mesh, export_swc, import_mesh, import_swc, load, save
from .ops.neuropils import distance_from_neuropil_surface, neuropil_point_depth
from .ops.plotting import Viewer, build_3d_subtree, plot_2d, plot_3d, plot_dendrogram
from .ops.tree_graphs import (
    breadth_first_iterator,
    breadth_first_search,
    check_reduced,
    count_branches,
    count_edges,
    count_leaves,
    count_nodes,
    count_roots,
    count_transitive_nodes,
    depth_first_iterator,
    depth_first_search,
    get_branch_indices,
    get_core_indices,
    get_degree_distribution,
    get_degrees,
    get_edge_coordinates,
    get_edge_indices,
    get_edge_length,
    get_leaf_indices,
    get_max_subtree_node,
    get_node_coordinates,
    get_node_depth,
    get_partition_asymmetry,
    get_post_order,
    get_root_coordinate,
    get_root_index,
    get_subtree,
    get_subtree_edge_coordinates,
    get_subtree_indices,
    get_subtree_node_coordinates,
    get_subtree_scores,
    get_total_cable_length,
    has_property,
    mask_subtree_from_root,
    reduce_tree,
    reroot_tree,
    update_reduced,
)
from .ops.units import (
    check_units_defined,
    convert_units,
    ensure_forest_units,
    get_units,
    get_voxel_spec,
    harmonize_forest_units,
    set_units,
    set_voxel_units,
)
from .utils.units import format_units_reference_table, list_unit_definitions

# Public API
__all__ = [
    "Tree",
    "Forest",
    "Tree_mesh",
    "Forest_mesh",
    "Neuropil",
    "Neuropils",
    "import_swc",
    "export_swc",
    "import_mesh",
    "export_mesh",
    "load",
    "save",
    "Viewer",
    "plot_2d",
    "plot_3d",
    "plot_dendrogram",
    "build_3d_subtree",
    "get_root_index",
    "get_leaf_indices",
    "get_branch_indices",
    "get_core_indices",
    "get_subtree_indices",
    "get_edge_indices",
    "count_roots",
    "count_nodes",
    "count_edges",
    "count_leaves",
    "count_branches",
    "count_transitive_nodes",
    "breadth_first_search",
    "breadth_first_iterator",
    "depth_first_search",
    "depth_first_iterator",
    "get_node_depth",
    "get_post_order",
    "get_node_coordinates",
    "get_root_coordinate",
    "get_subtree_node_coordinates",
    "get_edge_coordinates",
    "get_subtree_edge_coordinates",
    "check_reduced",
    "update_reduced",
    "has_property",
    "get_edge_length",
    "get_total_cable_length",
    "get_degrees",
    "get_degree_distribution",
    "reduce_tree",
    "reroot_tree",
    "mask_subtree_from_root",
    "get_subtree_scores",
    "get_max_subtree_node",
    "get_subtree",
    "get_partition_asymmetry",
    "distance_from_neuropil_surface",
    "neuropil_point_depth",
    "get_units",
    "get_voxel_spec",
    "set_units",
    "set_voxel_units",
    "convert_units",
    "check_units_defined",
    "harmonize_forest_units",
    "ensure_forest_units",
    "list_unit_definitions",
    "format_units_reference_table",
    "start_GUI",
    "reconstruct_neuropil_surface",
]

# Version
__version__ = "0.1.0"
