"""Functions for getting coordinates from trees"""

from typing import List, Tuple
from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    vertex_coordinates,
    vertex_coordinates_subtree,
    edge_coordinates,
    edge_coordinates_subtree,
)


def tree_node_coordinates(
    tree: _Tree, subset: int | List | None = None, SoA: bool = False
) -> ndarray:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    subset : int | List | None, optional
        _description_, by default None
    SoA : bool, optional
        _description_, by default False

    Returns
    -------
    ndarray
        _description_
    """
    return vertex_coordinates(tree.graph, subset, SoA)


def subtree_node_coordinates(
    tree: _Tree, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> ndarray:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    root : int
        _description_
    traversal_order : str, optional
        _description_, by default "Breadth"
    SoA : bool, optional
        _description_, by default False

    Returns
    -------
    ndarray
        _description_
    """
    return vertex_coordinates_subtree(tree.graph, root, traversal_order, SoA)


def tree_edge_coordinates(tree: _Tree, SoA: bool = False) -> Tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    SoA : bool, optional
        _description_, by default False

    Returns
    -------
    Tuple[ndarray, ndarray]
        _description_
    """
    return edge_coordinates(tree.graph, SoA)


def subtree_edge_coordinates(
    tree: _Tree, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> Tuple[ndarray, ndarray]:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    root : int
        _description_
    traversal_order : str, optional
        _description_, by default "Breadth"
    SoA : bool, optional
        _description_, by default False

    Returns
    -------
    Tuple[ndarray, ndarray]
        _description_
    """
    return edge_coordinates_subtree(tree.graph, root, traversal_order, SoA)
