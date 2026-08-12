"""Functions for distance calculations on trees."""

from numpy import ndarray

from ...core import _Tree
from ...utils.geometry_utils import euclidean_distance
from ...utils.graph_utils import (
    bind_edge_property,
    edge_coordinates,
    g_has_property,
)
from .tree_checks import check_reduced


def get_edge_length(tree: _Tree, bind: bool = True) -> ndarray | None:
    """Get length of edges.

    For a non-reduced neuron, this is the "path length" of each edge in the graph.
    For a reduced neuron, this is the Euclidean distance between nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, bind the lengths as an edge property and return None.
        If False, return the lengths array. By default True.

    Returns
    -------
    ndarray | None
        If bind=False, returns array of edge lengths. Otherwise returns None.
    """

    # check we don't have a length property alread
    if g_has_property(tree.graph, "Path_length", "e"):
        lengths = tree.graph.ep["Path_length"].a
    elif g_has_property(tree.graph, "Euclidean_length", "e"):
        lengths = tree.graph.ep["Euclidean_length"].a
    else:
        (x1, y1, z1), (x2, y2, z2) = edge_coordinates(tree.graph, SoA=True)
        lengths = euclidean_distance(x1, y1, z1, x2, y2, z2)

        if bind:
            # base property name on if this is a reduced graph
            p_name = "Euclidean_length" if check_reduced(tree) else "Path_length"
            bind_edge_property(
                tree.graph,
                property_name=p_name,
                property_dtype="double",
                property_data=lengths,
            )
    return lengths


def get_total_cable_length(tree: _Tree) -> float:
    """Compute total cable length of the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    float
        Total cable length (sum of all edge lengths).
    """
    # if we don't have path lengths add them
    if g_has_property(tree.graph, "Path_length", "e"):
        lengths = tree.graph.ep["Path_length"].a
    elif g_has_property(tree.graph, "Euclidean_length", "e"):
        lengths = tree.graph.ep["Euclidean_length"].a
    else:
        lengths = get_edge_length(tree, bind=False)

    return float(lengths.sum())
