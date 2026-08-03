"""Functions for spatial transformations on trees."""

from numpy import ndarray, array

from .coordinates import get_node_coordinates
from ...utils.geometry_utils.pca import eig_decomp
from ...utils.geometry_utils.rotations import compute_alignment_rotation, apply_rotation_steps
from ...utils.graph_utils.gt_properties import _set_coords_prop
from ...core import _Tree

def align_tree(
    tree: _Tree,
    robust: bool = True,
    b1: tuple = (0.0, 1.0, 0.0),
    b2: tuple = (1.0, 0.0, 0.0),
    b3: tuple = (0.0, 0.0, 0.1),
    bind: bool = True,
) -> ndarray | None:
    """PCA-align a single tree to canonical axes.

    Rotates node coordinates so the first three principal components align with
    *b1*, *b2*, and *b3*. Optionally writes rotated coordinates back to the
    tree graph.

    Parameters
    ----------
    tree : _Tree
        Tree to align.
    robust : bool, optional
        Use robust PCA. By default True.
    b1 : tuple, optional
        Target unit vector for PC1. By default (0.0, 1.0, 0.0).
    b2 : tuple, optional
        Target unit vector for PC2. By default (1.0, 0.0, 0.0).
    b3 : tuple, optional
        Target unit vector for PC3. By default (0.0, 0.0, 0.1).
    bind : bool, optional
        If True, write rotated coordinates to the graph and return None.
        By default True.

    Returns
    -------
    ndarray | None
        Rotated (N, 3) coordinates when bind=False; None when bind=True.
    """
    coords = get_node_coordinates(tree, SoA=True)
    coords -= coords.mean(axis=1, keepdims=True)
    x = coords[0]
    y = coords[1]
    z = coords[2]

    _, evecs = eig_decomp(x, y, z, robust=robust)

    ev1 = evecs[:, 0]
    ev2 = evecs[:, 1]
    ev3 = evecs[:, 2]

    # work out alignment steps
    step1, step2 = compute_alignment_rotation(ev1, ev2, ev3, b1, b2, b3)

    ### apply to x/y/z
    xr, yr, zr = apply_rotation_steps(x, y, z, step1, step2)

    if bind:
        ### set
        _set_coords_prop(tree.graph, array([xr, yr, zr]))
        return

    return array([xr, yr, zr]).T