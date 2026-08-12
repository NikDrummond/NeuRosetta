"""Least-squares shape fitting for pooled forest node coordinates."""

from vedo import Line, Plane, Sphere

from ...core import _Forest
from ...utils.vedo_utils import fit_circle, fit_line, fit_plane, fit_sphere


def fit_sphere_forest(forest: _Forest, return_obj: bool = False) -> Sphere | tuple:
    """Fit a sphere to pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    return_obj : bool, optional
        If True, return a vedo :class:`Sphere` object. If False, return
        ``(center, radius)``. By default False.

    Returns
    -------
    Sphere | tuple
        When *return_obj* is True, a vedo Sphere. Otherwise
        ``(center, radius)`` where *center* has shape ``(3,)``.
    """
    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    return fit_sphere(coords, return_obj=return_obj)


def fit_plane_forest(forest: _Forest, return_obj: bool = False) -> Plane | tuple:
    """Fit a plane to pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    return_obj : bool, optional
        If True, return a vedo :class:`Plane` object. If False, return
        ``(normal, center, variances)``. By default False.

    Returns
    -------
    Plane | tuple
        When *return_obj* is True, a vedo Plane. Otherwise
        ``(normal, center, variances)``.
    """
    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    return fit_plane(coords, return_obj=return_obj)


def fit_line_forest(forest: _Forest, return_obj: bool = False) -> Line | tuple:
    """Fit a 3D line to pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.
    return_obj : bool, optional
        If True, return a vedo :class:`Line` object. If False, return
        ``(slope, center, variances)``. By default False.

    Returns
    -------
    Line | tuple
        When *return_obj* is True, a vedo Line. Otherwise
        ``(slope, center, variances)``.
    """
    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    return fit_line(coords, return_obj=return_obj)


def fit_circle_forest(forest: _Forest) -> tuple:
    """Fit a circle to pooled node coordinates across a forest.

    Parameters
    ----------
    forest : _Forest
        Forest to analyse.

    Returns
    -------
    tuple
        ``(center, radius, normal_to_circle)``.
    """
    coords = forest.get_node_coordinates(SoA=False, merge_axis=0, show_progress=False)
    return fit_circle(coords)
