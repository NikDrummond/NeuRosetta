"""Functions for 3D geometric analysis of tree graphs."""

from typing import Tuple
from numpy import ndarray

from .counting import count_edges
from .coordinates import get_edge_coordinates

from ...core import _Tree
from ...utils.geometry_utils import normalize, align_with, planar_angle, signed_angle, angular_mean, angular_var, cross
from ...utils.graph_utils import set_property, g_has_property, get_property

def get_edge_angles(
    tree: _Tree,
    between_vector: Tuple | ndarray,
    perspective_vector: None | Tuple | ndarray = None,
    alignment_vector: None | Tuple | ndarray = None,
    degrees: bool = False,
    signed: bool = True,
    bind: bool = True,
) -> None | ndarray:
    """ """

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
        if g_has_property(tree.graph, "Edge_angles", "e"):
            c = False
        else:
            c = True
        set_property(
            tree.graph, "Edge_angles", edge_angles, "e", dtype="double", create=c
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
    """ """
    if weights is not None:
        assert count_edges(tree) == len(
            weights
        ), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    if g_has_property(tree.graph, "Edge_angles", "e"):
        edge_angles = get_property(tree.graph, "Edge_angles", "e")
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
    """ """
    if weights is not None:
        assert count_edges(tree) == len(
            weights
        ), f"If weights are given, len(weights) must be the same as number of edges ({count_edges(tree)}), not {len(weights)}"

    if g_has_property(tree.graph, "Edge_angles", "e"):
        edge_angles = get_property(tree.graph, "Edge_angles", "e")
    else:
        edge_angles = get_edge_angles(
            tree, between_vector, perspective_vector, alignment_vector, degrees, signed
        )

    return angular_var(edge_angles, weights)
