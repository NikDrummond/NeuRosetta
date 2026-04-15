"""functions for distance calculations on trees"""

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    bind_edge_property,
    edge_coordinates,
    g_has_property,
)

from ...utils.numpy_utils import pairwise_distance

from .tree_checks import check_reduced

# from ...utils.graph_utils.gt_properties import _bind_edge_property
# from .coordinates import edge_coordinates


def euclidean_edge_length(tree: _Tree, bind: bool = True) -> ndarray:
    """Get length of edges. If a non-reduced neuron is provided, this is the "path length" of each edge in the graph.
    If a reduced neuron is provided, this is the euclidean distance between nodes.

    Parameters
    ----------
    tree : Tree
        _description_
    subset : str | None, optional
        _description_, by default None
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    ndarray
        _description_
    """

    pairs = edge_coordinates(tree.graph)
    lengths = pairwise_distance(pairs[0], pairs[1])

    if bind:
        # base property name on if this is a reduced graph
        if check_reduced(tree):
            p_name = "Euclidean_length"
        else:
            p_name = "Path_length"
        bind_edge_property(
            tree.graph,
            property_name=p_name,
            property_dtype="double",
            property_data=lengths,
        )
        return None
    return lengths

def total_cable_length(tree: _Tree) -> float:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    float
        _description_
    """
    # if we don't have path lengths add them
    if g_has_property(tree.graph, "Path_length", "e"):
        lengths = tree.graph.ep['Path_length'].a
    elif g_has_property(tree.graph, "Euclidean_length", "e"):
        lengths = tree.graph.ep['Euclidean_length'].a
    else:
        lengths = euclidean_edge_length(tree, bind = False)

    return float(lengths.sum())
