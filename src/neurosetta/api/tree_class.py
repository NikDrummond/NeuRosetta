"""Tree class for neuron morphology analysis.

This module provides the Tree class which wraps the core _Tree class and
exposes a comprehensive API for neuron tree operations including traversal,
analysis, visualization, and I/O.
"""

from __future__ import annotations

from graph_tool.all import Graph

from ..core import _Tree
from ..io import (
    export_swc,
    save,
)
from ..ops.plotting import (
    plot_2d,
    plot_3d,
    plot_dendrogram,
)
from ..ops.tree_graphs import (
    align_coordinates,
    align_coordinates_to_vector,
    apply_rotation_steps_to_coordinates,
    breadth_first_iterator,
    breadth_first_search,
    center_coordinates_at_centroid,
    center_coordinates_at_root,
    check_reduced,
    coordinate_extent_along_axis,
    coordinate_mean_absolute_along_axis,
    coordinate_mean_along_axis,
    coordinate_minmax_along_axis,
    coordinate_pca,
    coordinate_projection_moments,
    coordinate_rms_along_axis,
    coordinate_std_along_axis,
    coordinate_variance_along_axis,
    count_bifurcations,
    count_branches,
    count_edges,
    count_leaves,
    count_nodes,
    count_roots,
    count_sections,
    count_transitive_nodes,
    depth_first_iterator,
    depth_first_search,
    fit_circle,
    fit_line,
    fit_plane,
    fit_sphere,
    get_bifurcation_angle_sums,
    get_bifurcation_angles,
    get_bifurcation_deihedral_beta,
    get_bifurcation_indices,
    get_branch_indices,
    get_convex_hull,
    get_convex_hull_volume,
    get_core_indices,
    get_degree_distribution,
    get_degrees,
    get_edge_angle_variance,
    get_edge_angles,
    get_edge_coordinates,
    get_edge_indices,
    get_edge_length,
    get_leaf_indices,
    get_max_subtree_node,
    get_mean_edge_angle,
    get_node_coordinates,
    get_node_depth,
    get_partition_asymmetry,
    get_post_order,
    get_radial_angle,
    get_root_coordinate,
    get_root_index,
    get_section_angular_deviation,
    get_subtree,
    get_subtree_edge_coordinates,
    get_subtree_indices,
    get_subtree_node_coordinates,
    get_subtree_scores,
    get_total_cable_length,
    mask_subtree_from_root,
    recenter_coordinates,
    reduce_tree,
    reroot_tree,
    rotate_coordinates,
    rotate_coordinates_about,
    scale_coordinates,
    scale_coordinates_about,
    scale_coordinates_along_pca,
    translate_coordinates,
    update_reduced,
)
from ..ops.units import (
    check_units_defined,
    convert_units,
    get_units,
    get_voxel_spec,
    set_units,
    set_voxel_units,
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

    get_bifurcation_indices = get_bifurcation_indices
    """Get the indices of bifurcation nodes"""

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

    count_sections = count_sections
    """Count the number of sections in a neuron (edges between root/branch/leaf nodes)"""

    count_bifurcations = count_bifurcations
    """Count the number of bifurcating nodes"""

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

    coordinate_mean_along_axis = coordinate_mean_along_axis
    """Compute mean projection of node coordinates onto an axis."""

    coordinate_variance_along_axis = coordinate_variance_along_axis
    """Compute variance of node coordinate projections onto an axis."""

    coordinate_std_along_axis = coordinate_std_along_axis
    """Compute standard deviation of node coordinate projections onto an axis."""

    coordinate_minmax_along_axis = coordinate_minmax_along_axis
    """Compute minimum and maximum projections onto an axis."""

    coordinate_extent_along_axis = coordinate_extent_along_axis
    """Compute extent of node coordinate projections onto an axis."""

    coordinate_rms_along_axis = coordinate_rms_along_axis
    """Compute root-mean-square projection onto an axis."""

    coordinate_mean_absolute_along_axis = coordinate_mean_absolute_along_axis
    """Compute mean absolute projection onto an axis."""

    coordinate_projection_moments = coordinate_projection_moments
    """Compute summary projection moments onto an axis."""

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
    get_edge_angles = get_edge_angles
    """Get the angle of edges relative to another vector and a perspective"""

    get_mean_edge_angle = get_mean_edge_angle
    """Get mean angle of edges relative to between_vector from perspective_vector."""

    get_edge_angle_variance = get_edge_angle_variance
    """Get angular vaiance of edges fom between vector fom perspecitve vector."""

    get_radial_angle = get_radial_angle
    """Get radial angle of edges from root node"""

    get_bifurcation_angles = get_bifurcation_angles
    """Get bifurcation angles (parent-child and child-child) at branch points."""

    get_bifurcation_angle_sums = get_bifurcation_angle_sums
    """ Get sum of edge angles at bifurcation points"""

    get_bifurcation_deihedral_beta = get_bifurcation_deihedral_beta
    """Get Dihedral beta angle for bifurcations (measure of planarity)"""

    # --- Shape Fitting ---
    fit_sphere = fit_sphere
    """Fit a sphere to neuron coordinates"""

    fit_line = fit_line
    """Fit a line to neuron coordinates"""

    fit_plane = fit_plane
    """Fit a plane to neuron coordinates"""

    fit_circle = fit_circle
    """Fit a circle to neuron coordinates"""

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

    get_section_angular_deviation = get_section_angular_deviation
    """Compute section angle mean and variance between core nodes for non-reduced neurons"""

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
    align_coordinates = align_coordinates
    """PCA-align node coordinates to canonical axes."""

    translate_coordinates = translate_coordinates
    """Translate all node coordinates by a fixed displacement."""

    center_coordinates_at_centroid = center_coordinates_at_centroid
    """Translate a tree so its node centroid lies at the origin."""

    center_coordinates_at_root = center_coordinates_at_root
    """Translate a tree so its root node lies at the origin."""

    recenter_coordinates = recenter_coordinates
    """Translate a tree so a specified point becomes the origin."""

    rotate_coordinates = rotate_coordinates
    """Rotate all node coordinates about an axis through the origin."""

    rotate_coordinates_about = rotate_coordinates_about
    """Rotate all node coordinates about an axis through a point."""

    scale_coordinates = scale_coordinates
    """Uniformly scale node coordinates about a point."""

    scale_coordinates_about = scale_coordinates_about
    """Scale node coordinates independently along each axis about a point."""

    align_coordinates_to_vector = align_coordinates_to_vector
    """Rotate a tree so a source direction aligns with a target direction."""

    scale_coordinates_along_pca = scale_coordinates_along_pca
    """Scale node coordinates along the tree PCA axes."""

    apply_rotation_steps_to_coordinates = apply_rotation_steps_to_coordinates
    """Apply two precomputed axis-angle rotations to all node coordinates."""

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
