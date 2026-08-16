"""Functions for distance calculations on trees."""

from typing import Literal

from numpy import ndarray

from ...core import _Tree
from ...utils.geometry_utils import euclidean_distance
from ...utils.graph_utils import (
    bind_edge_property,
    edge_coordinates,
    g_has_property,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import check_reduced

LengthType = Literal["Path", "Euclidean"]


def get_edge_length(tree: _Tree, bind: bool = True, recalculate: bool = False) -> ndarray | None:
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
    recalculate : bool, optional
        If True, recompute edge lengths from coordinates instead of reading
        existing edge properties. By default False.

    Returns
    -------
    ndarray | None
        If bind=False, returns array of edge lengths. Otherwise returns None.
    """

    lengths = None
    if not recalculate:
        if g_has_property(tree.graph, "Path_length", "e"):
            lengths = tree.graph.ep["Path_length"].a
        elif g_has_property(tree.graph, "Euclidean_length", "e"):
            lengths = tree.graph.ep["Euclidean_length"].a

    if lengths is None:
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


def get_total_cable_length(tree: _Tree, length_type: LengthType = "Path") -> float:
    """Compute total cable length of the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    length_type : {"Path", "Euclidean"}, optional
        Edge length property to sum. ``"Path"`` uses ``Path_length``;
        ``"Euclidean"`` uses ``Euclidean_length``. By default ``"Path"``.

    Returns
    -------
    float
        Total cable length (sum of all edge lengths).
    """
    prop_name = "Path_length" if length_type == "Path" else "Euclidean_length"

    if g_has_property(tree.graph, prop_name, "e"):
        lengths = tree.graph.ep[prop_name].a
    else:
        lengths = get_edge_length(tree, bind=False, recalculate=True)

    return float(lengths.sum())


enrich_tree_graph_docstrings(globals())
