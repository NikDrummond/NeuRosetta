"""Get verterx degree info from graphs"""

from typing import Tuple
from numpy import ndarray, unique
from graph_tool.all import Graph, EdgePropertyMap


def _check_deg_type_input(deg):

    accepted_degs = ["total", "in", "out"]
    if deg not in accepted_degs:
        raise ValueError(f"deg must be one of {accepted_degs}, not {deg}")


def get_vertex_degrees(
    g: Graph, deg: str = "total", weight: EdgePropertyMap | str | None = None
) -> ndarray:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_
    deg : str, optional
        _description_, by default "total"
    weight : EdgePropertyMap | str | None, optional
        _description_, by default None

    Returns
    -------
    ndarray
        _description_
    """
    _check_deg_type_input(deg)

    if isinstance(weight, str):
        weight = g.ep[weight]

    return g.degree_property_map(deg, weight).a


def degree_distribution(
    g: Graph, deg: str = "total", mass: bool = True
) -> Tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_
    deg : str, optional
        _description_, by default "total"
    mass : bool, optional
        _description_, by default True

    Returns
    -------
    Tuple[ndarray, ndarray]
        _description_
    """
    _check_deg_type_input(deg)

    degrees = get_vertex_degrees(g, deg)

    degrees, counts = unique(degrees, return_counts=True)

    if mass:
        counts = counts / counts.sum()

    return degrees, counts
