"""Backward-compatible facade for subtree helpers.

The implementations live in :mod:`neurosetta.utils.graph_utils.subtrees`.
Existing imports of ``subgraph_score``, ``max_subgraph_ind``,
``extract_subgraph``, and ``partition_asymmetry`` from this module keep
working.
"""

from __future__ import annotations

from .subtrees.asymmetry import partition_asymmetry
from .subtrees.extraction import extract_subgraph
from .subtrees.scoring import max_subgraph_ind, subgraph_score

__all__ = [
    "extract_subgraph",
    "max_subgraph_ind",
    "partition_asymmetry",
    "subgraph_score",
]
