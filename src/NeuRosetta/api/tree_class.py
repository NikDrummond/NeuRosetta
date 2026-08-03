"""Tree class for neuron morphology analysis.

This module provides the Tree class which wraps the core _Tree class and
exposes a comprehensive API for neuron tree operations including traversal,
analysis, visualization, and I/O.
"""
from __future__ import annotations

from graph_tool.all import Graph

from ..core import _Tree
from ..ops.tree_graphs import (
    get_root_index,
    get_leaf_indices,
    get_branch_indices,
    get_core_indices,
    get_subtree_indices,
    get_edge_indices,
    count_roots,
    count_nodes,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_nodes,
    breadth_first_search,
    breadth_first_iterator,
    depth_first_search,
    depth_first_iterator,
    get_node_depth,
    get_post_order,
    get_node_coordinates,
    get_root_coordinate,
    get_subtree_node_coordinates,
    get_edge_coordinates,
    get_subtree_edge_coordinates,
    coordinate_pca,
    get_convex_hull,
    get_convex_hull_volume,
    check_reduced,
    update_reduced,
    get_edge_length,
    get_total_cable_length,
    get_degrees,
    get_degree_distribution,
    reduce_tree,
    reroot_tree,
    mask_subtree_from_root,
    get_subtree_scores,
    get_max_subtree_node,
    get_subtree,
    get_partition_asymmetry,
    align_tree,
    get_mean_edge_angle,
    get_edge_angle_variance,
)

from ..ops.plotting import (
    plot_2d,
    plot_3d,
    plot_dendrogram,
)

from ..ops.units import (
    get_units,
    set_units,
    set_voxel_units,
    convert_units,
    check_units_defined,
    get_voxel_spec,
)

from ..io import (
    export_swc,
    save,
)

class Tree(_Tree):
    """Main class for single neuron tree operations.

    This class extends the core _Tree class and binds a comprehensive set
    of operations for neuron morphology analysis.

    Parameters
    ----------
    ID : int
        Unique identifier for the tree.
    metadata : dict
        Metadata dictionary containing file info, transformation history, etc.
    graph : graph_tool.Graph
        The underlying graph structure representing the neuron.

    Attributes
    ----------
    graph : graph_tool.Graph
        The neuron graph.
    metadata : dict
        Metadata dictionary (backed by ``graph.gp['metadata']``).
    ID : int
        Tree identifier (backed by ``graph.gp['ID']``).
    """

    __slots__ = ()

    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID=ID, metadata=metadata, graph=graph)

    # --- node indices ---
    get_root_index = get_root_index
    """Get the root node index."""

    get_leaf_indices = get_leaf_indices
    """Get indices of leaf nodes (out-degree == 0)."""

    get_branch_indices = get_branch_indices
    """Get indices of branch nodes (out-degree > 1)."""

    get_core_indices = get_core_indices
    """Get indices of core nodes (branches and leaves, optionally root)."""

    get_edge_indices = get_edge_indices
    """Get edge indices as (source, target) pairs."""

    get_subtree_indices = get_subtree_indices
    """Get node indices of subtree rooted at given vertex."""

    # --- counting ---
    count_roots = count_roots
    """Count root nodes."""

    count_nodes = count_nodes
    """Count total nodes."""

    count_edges = count_edges
    """Count total edges."""

    count_branches = count_branches
    """Count branch nodes."""

    count_leaves = count_leaves
    """Count leaf nodes."""

    count_transitive_nodes = count_transitive_nodes
    """Count transitive nodes (in-degree == out-degree == 1)."""

    # --- coordinates ---
    get_node_coordinates = get_node_coordinates
    """Get coordinates of all nodes."""

    get_root_coordinate = get_root_coordinate
    """Get coordinates of the root node"""

    get_subtree_node_coordinates = get_subtree_node_coordinates
    """Get coordinates of nodes in a subtree."""

    get_edge_coordinates = get_edge_coordinates
    """Get source and target coordinates for all edges."""

    get_subtree_edge_coordinates = get_subtree_edge_coordinates
    """Get source and target coordinates for edges in a subtree."""

    coordinate_pca = coordinate_pca
    """Perform Principal component analysis on Tree node coordinates"""

    get_convex_hull = get_convex_hull
    """Build a Convex Hull around neuron points using scipy.spatial.ConvexHull"""

    get_convex_hull_volume = get_convex_hull_volume
    """Get the convex hull volume around neuron points using scipy.spatial.ConvexHull"""

    # --- degrees ---
    get_degrees = get_degrees
    """Get degree array for all nodes."""

    get_degree_distribution = get_degree_distribution
    """Get degree distribution of the tree."""

    # --- distances ---
    get_edge_length = get_edge_length
    """Get Euclidean edge lengths."""

    get_total_cable_length = get_total_cable_length
    """Get total cable length of the tree."""

    # --- Geometry ---
    get_mean_edge_angle = get_mean_edge_angle
    """Get the mean angle of edges in Neuron between edges and between vector, from perspective_vector"""

    get_edge_angle_variance = get_edge_angle_variance
    """Get angular vaiance of edges fom between vector fom perspecitve vector."""

    # --- Traversals ---
    tree_breadth_first_search = breadth_first_search
    """Perform breadth-first search with custom visitor."""

    tree_breadth_first_iterator = breadth_first_iterator
    """Get BFS iterator or edge array."""

    tree_depth_first_search = depth_first_search
    """Perform depth-first search with custom visitor."""

    tree_depth_first_iterator = depth_first_iterator
    """Get DFS iterator or edge array."""

    get_post_order = get_post_order
    """Compute post-order traversal."""

    # --- Topological bits ---
    get_node_depth = get_node_depth
    """Compute node depths from root."""

    # --- Tree Surgery / editing ---
    get_reduced_tree = reduce_tree
    """Reduce tree by collapsing transitive vertices."""

    get_rerooted_tree = reroot_tree
    """Reroot tree at a new root vertex."""

    # --- Subtrees ---
    subtree_mask_from_root = mask_subtree_from_root
    """Create vertex/edge masks for subtree from root."""

    get_subtree_scores = get_subtree_scores
    """Compute subgraph scores for all vertices."""

    get_max_subtree_node = get_max_subtree_node
    """Get vertex index with maximum subgraph score."""

    get_subtree = get_subtree
    """Extract masked subtree as new tree."""

    get_partition_asymmetry = get_partition_asymmetry
    """Compute partition asymmetry at branch points."""

    # --- transformations ---
    align_tree = align_tree
    """Align neuron pca axis to basis"""

    # --- saving ---
    export_to_swc = export_swc
    """Export tree to SWC format."""

    save_tree = save
    """Save tree to file."""

    # --- plotting ---
    show_2d = plot_2d
    """Create 2D matplotlib plot of the tree."""

    show_3d = plot_3d
    """Create interactive 3D plot with vedo."""

    show_dendrogram = plot_dendrogram
    """Create dendrogram plot of the tree."""

    # --- checks ---
    is_reduced = check_reduced
    """Check if tree has no transitive vertices."""

    update_reduced = update_reduced
    """Update isReduced flag in metadata."""

    # --- units ---
    get_units = get_units
    """Get canonical spatial units from metadata."""

    set_units = set_units
    """Set spatial units, optionally converting geometry."""

    set_voxel_units = set_voxel_units
    """Tag coordinates as voxel indices with a cubic edge length."""

    get_voxel_spec = get_voxel_spec
    """Return voxel edge length metadata when units are voxels."""

    convert_units = convert_units
    """Convert coordinates and radii to target units."""

    check_units_defined = check_units_defined
    """Raise when tree units are dimensionless."""

    # has_property / list_properties / get_property / set_property /
    # del_property / revert_core_properties inherited from _Tree