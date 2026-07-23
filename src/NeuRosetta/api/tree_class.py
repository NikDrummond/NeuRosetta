"""Tree class for neuron morphology analysis.

This module provides the Tree class which wraps the core _Tree class and
exposes a comprehensive API for neuron tree operations including traversal,
analysis, visualization, and I/O.
"""
from __future__ import annotations

from graph_tool.all import Graph

from ..core import _Tree
from ..ops.tree_graphs import (
    get_root,
    get_leaves,
    get_branches,
    get_core_indices,
    get_subtree_nodes,
    get_edges,
    count_tree_roots,
    count_tree_nodes,
    count_tree_edges,
    count_tree_leaves,
    count_tree_branches,
    count_tree_transitive_nodes,
    breadth_first_search,
    breadth_first_iterator,
    depth_first_search,
    depth_first_iterator,
    compute_tree_depths,
    compute_post_order,
    tree_node_coordinates,
    subtree_node_coordinates,
    tree_edge_coordinates,
    subtree_edge_coordinates,
    check_reduced,
    update_reduced,
    tree_has_property,
    euclidean_edge_length,
    total_cable_length,
    get_node_degrees,
    tree_degree_distribution,
    reduce_tree,
    reroot_tree,
    mask_subtree_from_root,
    score_subtrees,
    max_subtree_ind,
    extract_subtree,
    node_partition_asymmetry,
)

from ..ops.plotting import (
    plot_2d,
    plot_3d,
    plot_dendrogram,
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
        Metadata dictionary.
    ID : int
        Tree identifier.
    """

    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID=ID, metadata=metadata, graph=graph)

    # --- node indices ---
    root_index = get_root
    """Get the root node index."""

    leaf_indices = get_leaves
    """Get indices of leaf nodes (out-degree == 0)."""

    branch_indices = get_branches
    """Get indices of branch nodes (out-degree > 1)."""

    core_indices = get_core_indices
    """Get indices of core nodes (branches and leaves, optionally root)."""

    edge_indices = get_edges
    """Get edge indices as (source, target) pairs."""

    subtree_indices = get_subtree_nodes
    """Get node indices of subtree rooted at given vertex."""

    # --- counting ---
    count_roots = count_tree_roots
    """Count root nodes."""

    count_nodes = count_tree_nodes
    """Count total nodes."""

    count_edges = count_tree_edges
    """Count total edges."""

    count_branches = count_tree_branches
    """Count branch nodes."""

    count_leaves = count_tree_leaves
    """Count leaf nodes."""

    count_transitive_nodes = count_tree_transitive_nodes
    """Count transitive nodes (in-degree == out-degree == 1)."""

    # --- coordinates ---
    get_node_coordinates = tree_node_coordinates
    """Get coordinates of all nodes."""

    get_subtree_node_coordinates = subtree_node_coordinates
    """Get coordinates of nodes in a subtree."""

    get_edge_coordinates = tree_edge_coordinates
    """Get source and target coordinates for all edges."""

    get_subtree_edge_coordinates = subtree_edge_coordinates
    """Get source and target coordinates for edges in a subtree."""

    # --- degrees ---
    get_degree_array = get_node_degrees
    """Get degree array for all nodes."""

    get_degree_distribution = tree_degree_distribution
    """Get degree distribution of the tree."""

    # --- distances ---
    get_edge_lengths = euclidean_edge_length
    """Get Euclidean edge lengths."""

    get_cable_length = total_cable_length
    """Get total cable length of the tree."""

    # --- Traversals ---
    tree_breadth_first_search = breadth_first_search
    """Perform breadth-first search with custom visitor."""

    tree_breadth_first_iterator = breadth_first_iterator
    """Get BFS iterator or edge array."""

    tree_depth_first_search = depth_first_search
    """Perform depth-first search with custom visitor."""

    tree_depth_first_iterator = depth_first_iterator
    """Get DFS iterator or edge array."""

    get_post_order_traversal = compute_post_order
    """Compute post-order traversal."""

    # --- Topological bits ---
    get_node_depths = compute_tree_depths
    """Compute node depths from root."""

    # --- Tree Surgery / editing ---
    get_reduced_tree = reduce_tree
    """Reduce tree by collapsing transitive vertices."""

    get_rerooted_tree = reroot_tree
    """Reroot tree at a new root vertex."""

    # --- Subtrees ---
    subtree_mask_from_root = mask_subtree_from_root
    """Create vertex/edge masks for subtree from root."""

    get_subtree_scores = score_subtrees
    """Compute subgraph scores for all vertices."""

    get_max_subtree_index = max_subtree_ind
    """Get vertex index with maximum subgraph score."""

    convert_to_subtree = extract_subtree
    """Extract masked subtree as new tree."""

    get_node_partition_asymmetry = node_partition_asymmetry
    """Compute partition asymmetry at branch points."""

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

    check_property = tree_has_property
    """Check if tree has a specific internal property."""