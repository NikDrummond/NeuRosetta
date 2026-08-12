"""Projection moments of pooled forest node coordinates onto an axis."""

from numpy import ndarray

from ...core import _Forest
from ...utils.geometry_utils.points import (
    extent_along_axis,
    mean_absolute_along_axis,
    mean_along_axis,
    minmax_along_axis,
    projection_moments,
    rms_along_axis,
    std_along_axis,
    variance_along_axis,
)


def _pooled_coordinate_arrays(
    forest: _Forest,
    axis: tuple | ndarray,
) -> tuple[ndarray, ndarray, ndarray, float, float, float]:
    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    ax, ay, az = axis
    return coords[:, 0], coords[:, 1], coords[:, 2], float(ax), float(ay), float(az)


def coordinate_mean_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the mean projection of pooled node coordinates onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Population mean projection onto *axis* across all nodes in the
        forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return mean_along_axis(x, y, z, ax, ay, az)


def coordinate_variance_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the variance of pooled node coordinate projections onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Population variance of projections onto *axis* across all nodes in
        the forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return variance_along_axis(x, y, z, ax, ay, az)


def coordinate_std_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the standard deviation of pooled projections onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Standard deviation of projections onto *axis* across all nodes in
        the forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return std_along_axis(x, y, z, ax, ay, az)


def coordinate_minmax_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> tuple[float, float]:
    """Compute minimum and maximum pooled projections onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    pmin : float
        Minimum projection onto *axis* across all nodes in the forest.
    pmax : float
        Maximum projection onto *axis* across all nodes in the forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return minmax_along_axis(x, y, z, ax, ay, az)


def coordinate_extent_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the extent of pooled node coordinate projections onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Difference between maximum and minimum projections onto *axis*
        across all nodes in the forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return extent_along_axis(x, y, z, ax, ay, az)


def coordinate_rms_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the root-mean-square pooled projection onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Root-mean-square projection onto *axis* across all nodes in the
        forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return rms_along_axis(x, y, z, ax, ay, az)


def coordinate_mean_absolute_along_axis_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> float:
    """Compute the mean absolute pooled projection onto an axis.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

    Returns
    -------
    float
        Mean absolute projection onto *axis* across all nodes in the forest.
    """
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return mean_absolute_along_axis(x, y, z, ax, ay, az)


def coordinate_projection_moments_forest(
    forest: _Forest,
    axis: tuple | ndarray,
) -> tuple[float, float, float, float, float]:
    """Compute summary projection moments of pooled node coordinates.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    axis : tuple | ndarray
        Axis direction ``(ax, ay, az)``. Assumed to be a unit vector.

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
    x, y, z, ax, ay, az = _pooled_coordinate_arrays(forest, axis)
    return projection_moments(x, y, z, ax, ay, az)
