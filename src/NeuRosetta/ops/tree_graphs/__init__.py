"""Tree graph operations."""

from .vertex_inds import (
    get_root,
    get_leaves,
    get_branches,
    get_core_indices,
    get_subtree_nodes,
    get_edges
)

from .counting import (
    count_tree_roots,
    count_tree_nodes,
    count_tree_edges,
    count_tree_leaves,
    count_tree_branches,
    count_tree_transitive_nodes
)

from .traversals import (
    breadth_first_search,
    breadth_first_iterator,
    depth_first_search,
    depth_first_iterator,
    compute_tree_depths,
    compute_post_order
)

from .coordinates import (
    tree_node_coordinates,
    subtree_node_coordinates,
    tree_edge_coordinates,
    subtree_edge_coordinates
)

from .tree_checks import (
    check_reduced,
    update_reduced,
    tree_has_property,
)

from .path_lengths import (
    euclidean_edge_length,
    total_cable_length,
)

from .degrees import (
    get_node_degrees,
    tree_degree_distribution,
)

from .tree_editing import (
    reduce_tree,
    reroot_tree,
)

from .subtrees import(
    mask_subtree_from_root,
    score_subtrees,
    max_subtree_ind,
)

__all__ = [
    "get_root",
    "get_leaves",
    "get_branches",
    "get_core_indices",
    "get_subtree_nodes",
    "get_edges",
    "count_tree_roots",
    "count_tree_nodes",
    "count_tree_edges",
    "count_tree_leaves",
    "count_tree_branches",
    "count_tree_transitive_nodes",
    "breadth_first_search",
    "breadth_first_iterator",
    "depth_first_search",
    "depth_first_iterator",
    "compute_tree_depths",
    "compute_post_order",
    "tree_node_coordinates",
    "subtree_node_coordinates",
    "tree_edge_coordinates",
    "subtree_edge_coordinates",
    "check_reduced",
    "update_reduced",
    "tree_has_property",
    "euclidean_edge_length",
    "total_cable_length",
    "get_node_degrees",
    "tree_degree_distribution",
    "reduce_tree",
    "reroot_tree",
    "mask_subtree_from_root",
    "score_subtrees",
    "max_subtree_ind",
]
