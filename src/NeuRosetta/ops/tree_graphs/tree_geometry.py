"""Functions for 3D geometric analysis of tree graphs."""

from typing import Tuple
from numpy import ndarray

from .counting import count_edges
from ...core import _Tree
from ...utils.geometry_utils import normalize, align_with, signed_angle, angular_mean, angular_var


def get_mean_edge_angle(tree: _Tree, 
                        between_vector: Tuple | ndarray,
                        perspective_vector: Tuple | ndarray,
                        alignment_vector: None | Tuple | ndarray = None,
                        weights: None | ndarray = None,
                        ) -> float:
    """Compute the circular mean of edge angles relative to a reference direction.

    For each edge, the signed angle is measured in the plane perpendicular to
    *perspective_vector* between the normalized edge direction and
    *between_vector*. The mean is computed with :func:`angular_mean`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray
        Look vector defining the plane normal for signed angles.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction before
        measuring angles. By default None.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular mean edge angle in radians.
    """
    if weights is not None:
        assert count_edges(tree) == len(weights), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    # subtract source from target and normalize
    starts, stops = tree.get_edge_coordinates(SoA = False)
    vectors = stops - starts
    xv,yv,zv = vectors.T
    xv,yv,zv = normalize(xv,yv, zv)
    # make sure perspective normal is aligned globaly (conserves sign, not always needed)
    if alignment_vector is not None:
        normal = align_with(*perspective_vector, *alignment_vector)
    else:
        normal = perspective_vector
    # get angles
    edge_angles = signed_angle(xv,yv,zv,*between_vector,*normal)

    return angular_mean(edge_angles, weights)

def get_edge_angle_variance(tree: _Tree, 
                            between_vector: Tuple | ndarray,
                            perspective_vector: Tuple | ndarray,
                            alignment_vector: None | Tuple | ndarray = None,
                            weights: None | ndarray = None) -> float:
    """Compute the circular variance of edge angles relative to a reference direction.

    For each edge, the signed angle is measured in the plane perpendicular to
    *perspective_vector* between the normalized edge direction and
    *between_vector*. The variance is computed with :func:`angular_var`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray
        Look vector defining the plane normal for signed angles.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction before
        measuring angles. By default None.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular variance of edge angles in radians.
    """
    if weights is not None:
        assert count_edges(tree) == len(weights), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    # subtract source from target and normalize
    starts, stops = tree.get_edge_coordinates(SoA = False)
    vectors = stops - starts
    xv,yv,zv = vectors.T
    xv,yv,zv = normalize(xv,yv, zv)
    # make sure perspective normal is aligned globaly (conserves sign, not always needed)
    if alignment_vector is not None:
        normal = align_with(*perspective_vector, *alignment_vector)
    else:
        normal = perspective_vector
    # get angles
    edge_angles = signed_angle(xv,yv,zv,*between_vector,*normal)

    return angular_var(edge_angles, weights)