"""Least-squares geometric shape fitting using vedo."""

from numpy import ndarray
from vedo import (
    Line,
    Plane,
    Sphere,
)
from vedo import (
    fit_circle as _fit_circle,
)
from vedo import (
    fit_line as _fit_line,
)
from vedo import (
    fit_plane as _fit_plane,
)
from vedo import (
    fit_sphere as _fit_sphere,
)


def fit_sphere(points: ndarray, return_obj: bool = False) -> Sphere | tuple:
    """Fit a sphere to a set of 3D points.

    Parameters
    ----------
    points : ndarray
        Point coordinates with shape ``(N, 3)``.
    return_obj : bool, optional
        If True, return a vedo :class:`Sphere` object. If False, return
        numeric parameters. By default False.

    Returns
    -------
    Sphere | tuple
        When *return_obj* is True, a vedo Sphere. Otherwise
        ``(center, radius)`` where *center* has shape ``(3,)``.
    """
    sphere = _fit_sphere(points)
    if return_obj:
        return sphere
    return sphere.center, sphere.radius


def fit_line(points: ndarray, return_obj: bool = False) -> Line | tuple:
    """Fit a 3D line to a set of points.

    Parameters
    ----------
    points : ndarray
        Point coordinates with shape ``(N, 3)``.
    return_obj : bool, optional
        If True, return a vedo :class:`Line` object. If False, return
        numeric parameters. By default False.

    Returns
    -------
    Line | tuple
        When *return_obj* is True, a vedo Line. Otherwise
        ``(slope, center, variances)``.
    """
    line = _fit_line(points)
    if return_obj:
        return line
    return line.slope, line.center, line.variances


def fit_plane(points: ndarray, return_obj: bool = False) -> Plane | tuple:
    """Fit a plane to a set of 3D points.

    Parameters
    ----------
    points : ndarray
        Point coordinates with shape ``(N, 3)``.
    return_obj : bool, optional
        If True, return a vedo :class:`Plane` object. If False, return
        numeric parameters. By default False.

    Returns
    -------
    Plane | tuple
        When *return_obj* is True, a vedo Plane. Otherwise
        ``(normal, center, variances)``.
    """
    plane = _fit_plane(points)
    if return_obj:
        return plane
    return plane.normal, plane.center, plane.variances


def fit_circle(points: ndarray) -> tuple:
    """Fit a circle to a set of 3D points.

    Parameters
    ----------
    points : ndarray
        Point coordinates with shape ``(N, 3)``.

    Returns
    -------
    tuple
        (center, radius, normal_to_circle).
    """
    return _fit_circle(points)
