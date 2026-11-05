"""Path coordinates, endpoint distance, and tortuosity."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils.paths import (
    path_coordinates,
    path_euclidean_distance,
    path_tortuosity,
)
from .._doc_helpers import enrich_tree_graph_docstrings


def get_path_coordinates(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return coordinates of the nodes on the path.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with ``x``, ``y``, and ``z``.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Shape ``(n_nodes, 3)`` in walk order. ``None`` when the path does
        not exist. No points are interpolated.
    """
    return path_coordinates(tree.graph, source, target, directed=directed)


def get_path_euclidean_distance(tree: _Tree, source: int, target: int) -> float:
    """Return the straight-line distance between two node coordinates.

    This is endpoint displacement. It does not sum ``Euclidean_length``
    and it does not depend on edge direction.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with coordinates.
    source, target : int
        Endpoint indices.

    Returns
    -------
    float
        Euclidean distance. 0 when both arguments are the same node.
    """
    return path_euclidean_distance(tree.graph, source, target)


def get_path_tortuosity(
    tree: _Tree,
    source: int,
    target: int,
    ep: str = "Path_length",
    *,
    directed: bool = True,
):
    """Return cable length divided by endpoint displacement.

    The default cable is ``Path_length``. ``Euclidean_length`` is not
    substituted. On a reduced tree the numerator is the stored section
    cable and the denominator is the straight line between the current
    endpoints.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with coordinates and the requested length property.
    source, target : int
        Path endpoints.
    ep : str, optional
        Length property. By default ``"Path_length"``.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    float or None
        Tortuosity. ``None`` when the directed path does not exist.
        ``nan`` when the endpoints coincide in space or are the same node.
    """
    return path_tortuosity(tree.graph, source, target, ep, directed=directed)


enrich_tree_graph_docstrings(globals())
