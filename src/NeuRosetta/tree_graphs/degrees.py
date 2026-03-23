from numpy import ndarray, unique
from graph_tool.all import EdgePropertyMap

from ..core import _Tree


def get_degrees(
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

    if isinstance(weight, str):
        weight = tree.graph.ep[weight]

    return tree.graph.degree_property_map(deg, weight).a

def degree_distribution(tree: _Tree, p: bool = True, deg: str = 'out', weight: EdgePropertyMap | str | None = None):
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    p : bool, optional
        _description_, by default True
    deg : str, optional
        _description_, by default 'out'
    weight : EdgePropertyMap | str | None, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    deg = get_degrees(tree, deg, weight)

    degs, counts = unique(deg, return_counts = True)

    if p:
        counts = counts / counts.sum()
    
    return degs, counts