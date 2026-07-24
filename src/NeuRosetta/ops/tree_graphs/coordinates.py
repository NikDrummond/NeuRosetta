"""Functions for getting coordinates from trees."""

from typing import List, Tuple
from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    vertex_coordinates,
    vertex_coordinates_subtree,
    edge_coordinates,
    edge_coordinates_subtree,
)


def get_node_coordinates(
    tree: _Tree, subset: int | List | None = None, SoA: bool = False
) -> ndarray:
    """Get coordinates of tree nodes.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    subset : int | List | None, optional
        Subset of node indices if only a subset of coordinates is wanted.
        If int, treated as a single index. If list, treated as array of indices.
        By default None (all vertices).
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions (N, 3). By default False.

    Returns
    -------
    ndarray
        Array of vertex coordinates with shape (3, N) if SoA=True,
        or (N, 3) if SoA=False, where N is the number of vertices.
    """
    return vertex_coordinates(tree.graph, subset, SoA)


def get_subtree_node_coordinates(
    tree: _Tree, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> ndarray:
    """Get coordinates of nodes in a subtree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Root vertex index defining the subtree.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth".
        By default "Breadth".
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions. By default False.

    Returns
    -------
    ndarray
        Array of vertex coordinates in subtree with shape (3, N) if SoA=True,
        or (N, 3) if SoA=False, where N is the number of vertices in the subtree.
    """
    return vertex_coordinates_subtree(tree.graph, root, traversal_order, SoA)


def get_edge_coordinates(tree: _Tree, SoA: bool = False) -> Tuple[ndarray, ndarray]:
    """Get source and target coordinates for all edges.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions. By default False.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Source and target coordinate arrays over edges, each with shape
        (3, E) if SoA=True, or (E, 3) if SoA=False, where E is the number
        of edges.
    """
    return edge_coordinates(tree.graph, SoA)


def get_subtree_edge_coordinates(
    tree: _Tree, root: int, traversal_order: str = "Breadth", SoA: bool = False
) -> Tuple[ndarray, ndarray]:
    """Get source and target coordinates for edges in a subtree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Root vertex index defining the subtree.
    traversal_order : str, optional
        Traversal order for subtree extraction, must be "Breadth" or "Depth".
        By default "Breadth".
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions. By default False.

    Returns
    -------
    Tuple[ndarray, ndarray]
        Source and target coordinate arrays over edges in subtree, each with
        shape (3, E) if SoA=True, or (E, 3) if SoA=False, where E is the
        number of edges in the subtree.
    """
    return edge_coordinates_subtree(tree.graph, root, traversal_order, SoA)