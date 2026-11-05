"""Subtree measurements on directed rooted trees.

Counts, cable, depth, Strahler order, subgraph score, partition asymmetry,
and subtree extraction each live in their own module. The bottom-up walk is
:mod:`._accumulate` and is not part of the public import surface.

Importing :mod:`neurosetta.utils.graph_utils.subgraphs` still exposes
``subgraph_score``, ``max_subgraph_ind``, ``extract_subgraph``, and
``partition_asymmetry``.
"""

from __future__ import annotations

from .asymmetry import partition_asymmetry
from .counts import (
    subgraph_branch_count,
    subgraph_edge_count,
    subgraph_leaf_count,
    subgraph_vertex_count,
)
from .depth import (
    subgraph_branch_depth,
    subgraph_depth,
    subgraph_height,
    subgraph_mean_leaf_distance,
)
from .extraction import extract_subgraph
from .hierarchy import max_strahler_order, strahler_order
from .lengths import subgraph_path_length
from .scoring import max_subgraph_ind, subgraph_score

__all__ = [
    "extract_subgraph",
    "max_strahler_order",
    "max_subgraph_ind",
    "partition_asymmetry",
    "strahler_order",
    "subgraph_branch_count",
    "subgraph_branch_depth",
    "subgraph_depth",
    "subgraph_edge_count",
    "subgraph_height",
    "subgraph_leaf_count",
    "subgraph_mean_leaf_distance",
    "subgraph_path_length",
    "subgraph_score",
    "subgraph_vertex_count",
]
