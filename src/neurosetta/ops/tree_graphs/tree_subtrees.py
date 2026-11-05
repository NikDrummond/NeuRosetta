"""Compatibility facade for subtree operations.

Implementations live in the ``tree_subtree_*`` modules. Historical imports
from this module keep working.
"""

from __future__ import annotations

from .tree_partition_asymmetry import get_partition_asymmetry
from .tree_subtree_scores import get_max_subtree_node, get_subtree_scores
from .tree_subtree_selection import get_subtree, mask_subtree_from_root

__all__ = [
    "get_max_subtree_node",
    "get_partition_asymmetry",
    "get_subtree",
    "get_subtree_scores",
    "mask_subtree_from_root",
]
