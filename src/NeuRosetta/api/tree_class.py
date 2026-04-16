"""Tree class"""

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
)

from ..ops.plotting import (
    plot_2d,
    plot_3d,
    plot_dendrogram
)

from ..io import (
    export_swc,
    save,
)

class Tree(_Tree):
    """_summary_

    Parameters
    ----------
    _Tree : _type_
        _description_
    """
    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID=ID, metadata=metadata, graph=graph)

    ### get indices
    root_index = get_root
    leaf_indices = get_leaves
    branch_indices = get_branches
    core_indices = get_core_indices
    edge_indices = get_edges
    subtree_indices = get_subtree_nodes

    ### counting
    count_roots = count_tree_roots
    count_nodes = count_tree_nodes
    count_edges = count_tree_edges
    count_branches = count_tree_branches
    count_leaves = count_tree_leaves
    count_transitive_nodes = count_tree_transitive_nodes

    ### coordinates
    get_node_coordinates = tree_node_coordinates
    get_subtree_node_coordinates = subtree_node_coordinates
    get_edge_coordinates = tree_edge_coordinates
    get_subtree_edge_coordinates = subtree_edge_coordinates

    ### degrees
    get_degree_array = get_node_degrees
    get_degree_distribution = tree_degree_distribution

    ### distances
    get_edge_lengths = euclidean_edge_length
    get_cable_length = total_cable_length

    ### Traversals
    tree_breadth_first_search = breadth_first_search
    tree_breadth_first_iterator = breadth_first_iterator
    tree_depth_first_search = depth_first_search
    tree_depth_first_iterator = depth_first_iterator

    get_post_order_traversal = compute_post_order

    ### Topological bits
    get_node_depths = compute_tree_depths

    ### Tree Surgery / editing
    get_reduced_tree = reduce_tree
    get_rerooted_tree = reroot_tree

    ### Subtrees
    subtree_mask_from_root = mask_subtree_from_root
    get_subtree_scores = score_subtrees
    get_max_subtree_index = max_subtree_ind
    convert_to_subtree = extract_subtree

    # saving
    export_to_swc = export_swc
    save_tree = save

    # plotting
    show_2d = plot_2d
    show_3d = plot_3d
    show_dendrogram = plot_dendrogram

    # checks
    is_reduced = check_reduced
    update_reduced = update_reduced
    check_property = tree_has_property

    # list properties (temp implementation)
    def list_properties(self):
        """List internal (bound) graph properties"""
        self.graph.list_properties()
