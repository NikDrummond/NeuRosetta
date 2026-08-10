"""Projection moments of tree node coordinates onto an axis."""

from typing import List, Tuple
from numpy import ndarray

from .coordinates import get_node_coordinates
from ...core import _Tree
from ...utils.geometry_utils.points import (
    mean_along_axis,
    variance_along_axis,
    std_along_axis,
    minmax_along_axis,
    extent_along_axis,
    rms_along_axis,
    mean_absolute_along_axis,
    projection_moments,
)


def _node_coordinate_arrays(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> Tuple[ndarray, ndarray, ndarray, float, float, float]:
    x, y, z = get_node_coordinates(tree, subset=subset, SoA=True)
    ax, ay, az = axis
    return x, y, z, float(ax), float(ay), float(az)


def coordinate_mean_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the mean projection of node coordinates onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Population mean projection onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return mean_along_axis(x, y, z, ax, ay, az)


def coordinate_variance_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the variance of node coordinate projections onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Population variance of projections onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return variance_along_axis(x, y, z, ax, ay, az)


def coordinate_std_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the standard deviation of node coordinate projections onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Standard deviation of projections onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return std_along_axis(x, y, z, ax, ay, az)


def coordinate_minmax_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> Tuple[float, float]:
    """Compute minimum and maximum projections onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    pmin : float
        Minimum projection onto *axis*.
    pmax : float
        Maximum projection onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return minmax_along_axis(x, y, z, ax, ay, az)


def coordinate_extent_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the extent of node coordinate projections onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Difference between maximum and minimum projections onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return extent_along_axis(x, y, z, ax, ay, az)


def coordinate_rms_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the root-mean-square projection onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Root-mean-square projection onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return rms_along_axis(x, y, z, ax, ay, az)


def coordinate_mean_absolute_along_axis(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> float:
    """Compute the mean absolute projection onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    float
        Mean absolute projection onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return mean_absolute_along_axis(x, y, z, ax, ay, az)


def coordinate_projection_moments(
    tree: _Tree,
    axis: Tuple | ndarray,
    subset: int | List | None = None,
) -> Tuple[float, float, float, float, float]:
    """Compute summary projection moments of node coordinates onto an axis.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.
    subset : int | list | None, optional
        Subset of node indices. By default None (all vertices).

    Returns
    -------
    mean : float
        Population mean projection onto *axis*.
    variance : float
        Population variance of projections onto *axis*.
    std : float
        Standard deviation of projections onto *axis*.
    pmin : float
        Minimum projection onto *axis*.
    pmax : float
        Maximum projection onto *axis*.
    """
    x, y, z, ax, ay, az = _node_coordinate_arrays(tree, axis, subset)
    return projection_moments(x, y, z, ax, ay, az)
