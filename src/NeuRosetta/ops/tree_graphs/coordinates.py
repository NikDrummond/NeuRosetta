"""Functions for getting coordinates from trees."""

from typing import List, Tuple
from numpy import ndarray
from scipy.spatial import ConvexHull

from ...core import _Tree
from ...utils.graph_utils import (
    vertex_coordinates,
    vertex_coordinates_subtree,
    edge_coordinates,
    edge_coordinates_subtree,
)
from ...utils.geometry_utils import (
    eig_decomp
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

def get_root_coordinate(
    tree: _Tree, SoA: bool = False
) -> ndarray:
    """Get coordinates of the tree root node.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    SoA : bool, optional
        Whether to return in Structure of Arrays format (dimensions by vertices).
        If False, return vertices by dimensions (N, 3). By default False.

    Returns
    -------
    ndarray
        Array of vertex coordinates with shape (3, N) if SoA=True,
        or (N, 3) if SoA=False, where N is the number of roots.
    """
    return vertex_coordinates(tree.graph, subset = tree.get_root_index(), SoA = SoA)

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

def coordinate_pca(tree: _Tree, robust:bool = True, norm:bool = True) -> Tuple[ndarray,ndarray]:
    """Perform PCA on tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    robust : bool, optional
        Use robust covariance estimation. By default True.
    norm : bool, optional
        Normalize eigenvalues to sum to 1. By default True.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(evals, evecs)`` from :func:`eig_decomp`: eigenvalues in descending
        order and corresponding eigenvectors as columns.
    """
    x,y,z = get_node_coordinates(tree, SoA = True)
    return eig_decomp(x,y,z, robust = robust, norm = norm)

def get_convex_hull(tree:_Tree, bind:bool = False) -> ConvexHull | None:
    """Build a convex hull around tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, store the hull as graph property ``Convex_hull`` and return
        None. By default False.

    Returns
    -------
    ConvexHull | None
        SciPy convex hull when bind=False; None when bind=True.
    """
    cv = ConvexHull(get_node_coordinates(tree, SoA = False))
    if bind:
        tree.set_property('Convex_hull', cv, level  = 'g', dtype = 'object', create = True)
        return
    else: 
        return cv

def get_convex_hull_volume(tree:_Tree, bind:bool = False) -> float:
    """Return the volume enclosed by the convex hull of tree node coordinates.

    Reuses a cached ``Convex_hull`` graph property when present.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, compute and store the hull as graph property ``Convex_hull``
        before returning its volume. By default False.

    Returns
    -------
    float
        Convex hull volume in cubic tree units.
    """
    if tree.has_property('Convex_hull', level = 'g'):
        return tree.graph.gp['Convex_hull'].volume

    if bind:
        get_convex_hull(tree, bind = True)
        return tree.graph.gp['Convex_hull'].volume
    else:
        cv = get_convex_hull(tree, bind = False)
        return cv.volume
    