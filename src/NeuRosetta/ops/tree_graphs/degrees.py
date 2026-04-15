"""functions to get degree info from trees"""

from typing import Tuple
from numpy import ndarray
from graph_tool.all import EdgePropertyMap

from ...core import _Tree
from ...utils.graph_utils import get_vertex_degrees, degree_distribution


def get_node_degrees(
    tree: _Tree,
    deg: str = "total",
    weight: EdgePropertyMap | str | None = None,
) -> ndarray:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    deg : str, optional
        _description_, by default 'total'
    p : bool, optional
        _description_, by default False
    weight : EdgePropertyMap | str | None, optional
        _description_, by default None

    Returns
    -------
    ndarray
        _description_
    """

    return get_vertex_degrees(tree.graph, deg, weight)


def tree_degree_distribution(
    tree: _Tree, deg: str = "total", mass: bool = True
) -> Tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    deg : str, optional
        _description_, by default "total"
    mass : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    return degree_distribution(tree.graph, deg, mass)
