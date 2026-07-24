"""Tree graph operations."""

from .vertex_inds import (
    get_root_index,
    get_leaf_indices,
    get_branch_indices,
    get_core_indices,
    get_subtree_indices,
    get_edge_indices
)

from .counting import (
    count_roots,
    count_nodes,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_nodes
)

from .traversals import (
    breadth_first_search,
    breadth_first_iterator,
    depth_first_search,
    depth_first_iterator,
    get_node_depth,
    get_post_order
)

from .coordinates import (
    get_node_coordinates,
    get_subtree_node_coordinates,
    get_edge_coordinates,
    get_subtree_edge_coordinates
)

from .tree_checks import (
    check_reduced,
    update_reduced,
    has_property,
)

from .path_lengths import (
    get_edge_length,
    get_total_cable_length,
)

from .degrees import (
    get_degrees,
    get_degree_distribution,
)

from .tree_editing import (
    reduce_tree,
    reroot_tree,
)

from .subtrees import(
    mask_subtree_from_root,
    get_subtree_scores,
    get_max_subtree_node,
    get_subtree,
    get_partition_asymmetry,
)

__all__ = [
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
]
