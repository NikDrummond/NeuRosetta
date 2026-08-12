"""Get vertex degree information from graphs."""
from __future__ import annotations

from typing import Tuple
from numpy import ndarray, unique
from graph_tool.all import Graph, EdgePropertyMap


def _check_deg_type_input(deg: str) -> None:
    """Validate degree type input.

    Parameters
    ----------
    deg : str
        Degree type to validate.

    Raises
    ------
    ValueError
        If deg is not one of {"total", "in", "out"}.
    """
    accepted_degs = ["total", "in", "out"]
    if deg not in accepted_degs:
        raise ValueError(f"deg must be one of {accepted_degs}, not {deg}")


def get_vertex_degrees(
    g: Graph, deg: str = "total", weight: EdgePropertyMap | str | None = None
) -> ndarray:
    """Get vertex degrees for all vertices in a graph.

    Parameters
    ----------
    g : Graph
        Input graph.
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
    _check_deg_type_input(deg)

    if isinstance(weight, str):
        weight = g.ep[weight]

    return g.degree_property_map(deg, weight).a


def degree_distribution(
    g: Graph, deg: str = "total", mass: bool = True
) -> Tuple[ndarray, ndarray]:
    """Compute the degree distribution of a graph.

    Parameters
    ----------
    g : Graph
        Input graph.
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
    _check_deg_type_input(deg)

    degrees = get_vertex_degrees(g, deg)

    degrees, counts = unique(degrees, return_counts=True)

    if mass:
        counts = counts / counts.sum()

    return degrees, counts