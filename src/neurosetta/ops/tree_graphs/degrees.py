"""Functions to get degree information from trees."""
from __future__ import annotations

from typing import Tuple
from numpy import ndarray
from graph_tool.all import EdgePropertyMap

from ...core import _Tree
from ...utils.graph_utils import get_vertex_degrees, degree_distribution


def get_degrees(
    tree: _Tree,
    deg: str = "total",
    weight: EdgePropertyMap | str | None = None,
) -> ndarray:
    """Get vertex degrees for all nodes in a tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    deg : str, optional
        Degree type to compute. One of "total", "in", "out". By default "total".
    weight : EdgePropertyMap | str | None, optional
        Edge property map or property name to use as edge weights. If None,
        unweighted degrees are computed. By default None.

    Returns
    -------
    ndarray
        Array of vertex degrees with length equal to number of vertices.
    """
    return get_vertex_degrees(tree.graph, deg, weight)


def get_degree_distribution(
    tree: _Tree, deg: str = "total", mass: bool = True
) -> Tuple[ndarray, ndarray]:
    """Compute the degree distribution of a tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    deg : str, optional
        Degree type to compute. One of "total", "in", "out". By default "total".
    mass : bool, optional
        If True, normalize counts to sum to 1 (probability mass function).
        If False, return raw counts. By default True.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Tuple of (degrees, counts) where degrees are the unique degree values
        and counts are their frequencies (normalized if mass=True).
    """
    return degree_distribution(tree.graph, deg, mass)