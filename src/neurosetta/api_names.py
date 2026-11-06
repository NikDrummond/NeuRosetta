"""Public method names that differ from the underlying operation name.

This module is the single copy of those aliases. ``ops`` uses them when
linking docstrings. ``utils.metrics`` uses them when a registered metric
name is not the Tree method name. Neither layer should own the table.
"""

from __future__ import annotations

# Tree method name when it differs from the underlying op function name.
TREE_METHOD_ALIASES: dict[str, str] = {
    "breadth_first_search": "tree_breadth_first_search",
    "breadth_first_iterator": "tree_breadth_first_iterator",
    "depth_first_search": "tree_depth_first_search",
    "depth_first_iterator": "tree_depth_first_iterator",
    "check_reduced": "is_reduced",
    "reduce_tree": "get_reduced_tree",
    "reroot_tree": "get_rerooted_tree",
    "mask_subtree_from_root": "subtree_mask_from_root",
}

# Forest method name when it differs from the underlying op function name.
FOREST_METHOD_ALIASES: dict[str, str] = {
    "check_reduced": "is_reduced",
}
