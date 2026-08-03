"""Functions for global coordinate transformation of forests."""

from typing import Tuple
from numpy import array, ndarray
from scipy.spatial import ConvexHull

from ...core import _Forest
from ...utils.graph_utils.gt_properties import _set_coords_prop
from ...utils.geometry_utils.pca import eig_decomp
from ...utils.geometry_utils.rotations import compute_alignment_rotation, apply_rotation_steps
from .utils import _split_array_vertex, _split_inds

def _set_global_coords(forest: _Forest, x:ndarray,y:ndarray,z:ndarray):
    s_index = _split_inds(forest)

    x = _split_array_vertex(x, s_index)
    y = _split_array_vertex(y, s_index)
    z = _split_array_vertex(z, s_index)

    for i in range(len(forest)):
        c = array([x[i],y[i],z[i]])
        _set_coords_prop(forest[i].graph, c)

def align_forest(
    forest: _Forest,
    robust: bool = True,
    b1: tuple = (0.0, 1.0, 0.0),
    b2: tuple = (1.0, 0.0, 0.0),
    b3: tuple = (0.0, 0.0, 0.1),
    unpack: bool = False,
    bind: bool = True,
) -> ndarray | None:
    """PCA-align all trees in a forest to canonical axes.

    Rotates node coordinates so the first three principal components align with
    *b1*, *b2*, and *b3*. Optionally writes rotated coordinates back to each
    tree graph.

    Parameters
    ----------
    forest : _Forest
        Forest to align.
    robust : bool, optional
        Use robust PCA. By default True.
    b1 : tuple, optional
        Target unit vector for PC1. By default (0.0, 1.0, 0.0).
    b2 : tuple, optional
        Target unit vector for PC2. By default (1.0, 0.0, 0.0).
    b3 : tuple, optional
        Target unit vector for PC3. By default (0.0, 0.0, 0.1).
    unpack : bool, optional
        If True and bind=False, return per-tree coordinate arrays. By default
        False.
    bind : bool, optional
        If True, write rotated coordinates to forest graphs and return None.
        By default True.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray] | None
        Rotated (x, y, z) arrays when bind=False; None when bind=True.
    """

    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    coords -= coords.mean(axis=0, keepdims=True)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    _, evecs = eig_decomp(x, y, z, robust=robust)

    ev1 = evecs[:, 0]
    ev2 = evecs[:, 1]
    ev3 = evecs[:, 2]

    # work out alignment steps
    step1, step2 = compute_alignment_rotation(ev1, ev2, ev3, b1, b2, b3)

    ### apply to x/y/z
    xr, yr, zr = apply_rotation_steps(x, y, z, step1, step2)

    if bind:
        _set_global_coords(forest, xr, yr, zr)
        return

    if unpack:
        s_index = _split_inds(forest)
        xr = _split_array_vertex(xr, s_index)
        yr = _split_array_vertex(yr, s_index)
        zr = _split_array_vertex(zr, s_index)

    return xr, yr, zr

def forest_pca(forest: _Forest, robust: bool = False, norm:bool = True) -> Tuple[ndarray, ndarray]:
    """Perform PCA on pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    robust : bool, optional
        Use robust covariance estimation. By default False.
    norm : bool, optional
        Normalize eigenvalues to sum to 1. By default True.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(evals, evecs)`` from :func:`eig_decomp`: eigenvalues in descending
        order and corresponding eigenvectors as columns.
    """
    x,y,z = forest.get_node_coordinates(SoA = True, merge_axis = 1, show_progress = False)
    return eig_decomp(x,y,z, robust = robust, norm = norm)

def get_forest_convex_hull(forest:_Forest) -> ConvexHull:
    """Build a convex hull around pooled node coordinates in a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.

    Returns
    -------
    ConvexHull
        SciPy convex hull of all node coordinates.
    """
    return ConvexHull(forest.get_node_coordinates(SoA = False, merge_axis = 0, show_progress = False))

def get_forest_convex_hull_volume(forest:_Forest) -> float:
    """Return the volume enclosed by the convex hull of forest node coordinates.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.

    Returns
    -------
    float
        Convex hull volume in cubic tree units.
    """
    return ConvexHull(forest.get_node_coordinates(SoA = False, merge_axis = 0, show_progress = False)).volume
