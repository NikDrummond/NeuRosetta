"""Functions for 3D geometric analysis of tree graphs."""

from typing import Tuple
from numpy import ndarray

from .counting import count_edges
from .coordinates import get_edge_coordinates, get_root_coordinate

from ...core import _Tree
from ...utils.geometry_utils import (
    normalize,
    align_with,
    planar_angle,
    signed_angle,
    angular_mean,
    angular_var,
    cross,
    angle,
)
from ...utils.graph_utils import set_property, g_has_property, get_property


def get_edge_angles(
    tree: _Tree,
    between_vector: Tuple | ndarray,
    perspective_vector: None | Tuple | ndarray = None,
    alignment_vector: None | Tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    bind: bool = False,
) -> None | ndarray:
    """Compute the angle of each edge relative to a reference direction.

    For each edge, the angle is measured between the normalized edge direction
    and *between_vector* in the plane perpendicular to *perspective_vector*.
    When *perspective_vector* is None, it defaults to the cross product of
    each edge direction with *between_vector*.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None, optional
        Look vector defining the plane normal for angle measurement. When
        None, computed per edge as ``cross(edge, between_vector)``.
        By default None.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.
    signed : bool, optional
        If True, return signed planar angles via :func:`signed_angle`; if
        False, return unsigned angles via :func:`planar_angle`.
        By default True.
    bind : bool, optional
        If True, store results as edge property ``Edge_angle`` and return
        None. By default False.

    Returns
    -------
    ndarray | None
        Per-edge angles with length equal to the number of edges when
        bind=False; None when bind=True.
    """

    # subtract source from target and normalize
    starts, stops = get_edge_coordinates(tree, SoA=False)
    vectors = stops - starts
    xv, yv, zv = vectors.T
    xv, yv, zv = normalize(xv, yv, zv)

    # if no perspective vectoer given, use cross of between and edges
    if perspective_vector is None:
        perspective_vector = cross(xv, yv, zv, *between_vector)

    # If alignment vector is given make sure perspective normal is aligned globaly (conserves sign, not always needed)
    if alignment_vector is not None:
        normal = align_with(*perspective_vector, *alignment_vector)
    else:
        normal = perspective_vector

    # get angles
    if signed:
        edge_angles = signed_angle(xv, yv, zv, *between_vector, *normal, degrees)
    else:
        edge_angles = planar_angle(xv, yv, zv, *between_vector, *normal, degrees)

    if bind:
        if g_has_property(tree.graph, "Edge_angle", "e"):
            c = False
        else:
            c = True
        set_property(
            tree.graph, "Edge_angle", edge_angles, "e", dtype="double", create=c
        )
        return
    return edge_angles


def get_mean_edge_angle(
    tree: _Tree,
    between_vector: Tuple | ndarray,
    perspective_vector: None | Tuple | ndarray,
    alignment_vector: None | Tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    weights: None | ndarray = None,
) -> float:
    """Compute the circular mean of edge angles relative to a reference direction.

    Reuses a cached ``Edge_angle`` edge property when present; otherwise
    calls :func:`get_edge_angles`. The mean is computed with
    :func:`angular_mean`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None
        Look vector defining the plane normal for signed angles. Passed to
        :func:`get_edge_angles` when ``Edge_angle`` is not cached.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Return the mean in degrees instead of radians. By default False.
    signed : bool, optional
        Use signed planar angles when computing uncached edge angles.
        By default True.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular mean edge angle in radians (or degrees when *degrees* is
        True).
    """
    if weights is not None:
        assert count_edges(tree) == len(
            weights
        ), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    if g_has_property(tree.graph, "Edge_angle", "e"):
        edge_angles = get_property(tree.graph, "Edge_angle", "e")
    else:
        edge_angles = get_edge_angles(
            tree, between_vector, perspective_vector, alignment_vector, degrees, signed
        )

    return angular_mean(edge_angles, weights)


def get_edge_angle_variance(
    tree: _Tree,
    between_vector: Tuple | ndarray,
    perspective_vector: None | Tuple | ndarray = None,
    alignment_vector: None | Tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    weights: None | ndarray = None,
) -> float:
    """Compute the circular variance of edge angles relative to a reference direction.

    Reuses a cached ``Edge_angle`` edge property when present; otherwise
    calls :func:`get_edge_angles`. The variance is computed with
    :func:`angular_var`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    between_vector : tuple | ndarray
        Reference direction (3,) against which edge angles are measured.
    perspective_vector : tuple | ndarray | None, optional
        Look vector defining the plane normal for signed angles. Passed to
        :func:`get_edge_angles` when ``Edge_angle`` is not cached.
        By default None.
    alignment_vector : tuple | ndarray | None, optional
        If given, flip *perspective_vector* to align with this direction
        before measuring angles. By default None.
    degrees : bool, optional
        Interpret cached or computed angles in degrees. By default False.
    signed : bool, optional
        Use signed planar angles when computing uncached edge angles.
        By default True.
    weights : ndarray | None, optional
        Per-edge weights; length must equal the number of edges. By default
        None.

    Returns
    -------
    float
        Circular variance of edge angles in radians (or degrees when
        *degrees* is True).
    """
    if weights is not None:
        assert count_edges(tree) == len(
            weights
        ), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    if g_has_property(tree.graph, "Edge_angle", "e"):
        edge_angles = get_property(tree.graph, "Edge_angle", "e")
    else:
        edge_angles = get_edge_angles(
            tree, between_vector, perspective_vector, alignment_vector, degrees, signed
        )

    return angular_var(edge_angles, weights)


def get_radial_angle(
    tree: _Tree,
    alignment_vector: None | Tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    bind: bool = True,
) -> None | ndarray:
    """Compute the radial angle of each edge relative to the tree root.

    For each edge, measures the angle between the direction from the edge
    midpoint toward the target node and the direction from the root to the
    edge midpoint.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    alignment_vector : tuple | ndarray | None, optional
        Reference direction used to orient the sign when *signed* is True.
        Required when *signed* is True. By default None.
    degrees : bool, optional
        Return angles in degrees instead of radians. By default False.
    signed : bool, optional
        If True, return signed angles via :func:`signed_angle`; if False,
        return unsigned angles via :func:`angle`. By default True.
    bind : bool, optional
        If True, store results as edge property ``Radial_angle`` and return
        None. By default True.

    Returns
    -------
    ndarray | None
        Per-edge radial angles with length equal to the number of edges when
        bind=False; None when bind=True.

    Raises
    ------
    ValueError
        If *signed* is True and *alignment_vector* is None.
    """
    # source and target coords
    source, target = get_edge_coordinates(tree, SoA=True)
    # angle related data:
    vectors = target - source
    vectors = normalize(*vectors)
    section_mid_points = (source + target) / 2
    rc = get_root_coordinate(tree)
    center_vecs = section_mid_points - rc[:, None]
    center_vecs = normalize(*center_vecs)

    section_ends = target - section_mid_points
    section_ends = normalize(*section_ends)

    crosses = cross(*center_vecs, *section_ends)

    if signed:

        if alignment_vector is None:
            raise ValueError(f"alignment_vector must be given if signed is True")

        normals = normalize(*align_with(*crosses, *alignment_vector))
        a = signed_angle(*section_ends, *center_vecs, *normals, degrees=degrees)
    else:
        a = angle(*section_ends, *center_vecs, degrees=degrees)

    if bind:
        tree.set_property("Radial_angle", a, "e", dtype="double", create=True)
        return

    return a
