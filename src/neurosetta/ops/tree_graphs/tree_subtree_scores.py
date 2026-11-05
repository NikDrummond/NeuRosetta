"""Subtree score used to choose a subtree root."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import max_subgraph_ind, subgraph_score
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import has_property
from .tree_path_lengths import get_edge_length


def get_subtree_scores(tree: _Tree, bind: bool = True) -> ndarray | None:
    """Compute subgraph scores for all vertices in the tree.

    Branching nodes score ``1 - subtree_cable / total_cable + subtree_leaves / total_leaves``.
    Every other node scores 0. Cable prefers ``Path_length`` and falls back to
    ``Euclidean_length`` when ``Path_length`` is absent.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, bind the scores as a vertex property and return None.
        If False, return the score array. By default True.

    Returns
    -------
    ndarray | None
        Array of subgraph scores if bind=False, otherwise None.

    Notes
    -----
    If neither length property exists, straight-line lengths are computed and
    bound (``Path_length`` on an unreduced tree, ``Euclidean_length`` on a
    reduced tree). That fallback is specific to this historical score.
    """
    if not (has_property(tree, "Path_length", "e") or has_property(tree, "Euclidean_length", "e")):
        get_edge_length(tree, bind=True)

    score = subgraph_score(tree.graph, bind)
    return score if not bind else None


def get_max_subtree_node(tree: _Tree) -> int:
    """Get the vertex index with the maximum subgraph score.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Vertex index with the maximum subgraph score.
    """
    if not has_property(tree, "subgraph_score", "v"):
        get_subtree_scores(tree, bind=True)
    return max_subgraph_ind(tree.graph)


enrich_tree_graph_docstrings(globals())
