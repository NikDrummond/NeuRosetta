"""Subtree cable length on a NeuRosetta tree."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import subgraph_path_length
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_subtree_common import resolve_tree_edge_weights


def get_subtree_path_length(
    tree: _Tree,
    bind: bool = False,
    length_property: str = "Path_length",
) -> ndarray | None:
    """Sum edge lengths inside the subtree of every node.

    The parent edge of the subtree root is excluded. A leaf has length 0.
    The tree root's value is the total length of the chosen edge property.

    ``Path_length`` is physical cable, including lengths preserved when a
    tree was reduced. ``Euclidean_length`` is the straight-line length stored
    on each edge. One is not substituted for the other.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store ``subtree_path_length`` (or
        ``subtree_path_length_euclidean``) and return None. By default False.
        Computing a missing length does not write it onto the graph; pass the
        array through and leave existing properties untouched.
    length_property : {"Path_length", "Euclidean_length"}, optional
        Edge weights to sum. By default ``"Path_length"``.

    Returns
    -------
    ndarray or None
        Float length per node when ``bind`` is False.

    Notes
    -----
    A reduced tree without ``Path_length`` raises. An unreduced tree without
    ``Path_length`` uses the straight-line length of each uncollapsed segment,
    which is the physical length of that segment. That fallback is not used
    for reduced trees.
    """
    weights = resolve_tree_edge_weights(tree, length_property)
    return subgraph_path_length(
        tree.graph,
        bind=bind,
        length_property=length_property,
        weights=weights,
    )


enrich_tree_graph_docstrings(globals())
