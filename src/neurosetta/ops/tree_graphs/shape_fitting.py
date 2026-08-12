"""Least-squares shape fitting for neuron tree node coordinates."""

from typing import Tuple
from vedo import (
    Sphere,
    Line,
    Plane,
)

from ...core import _Tree
from ...utils.graph_utils import vertex_coordinates
from ...utils.vedo_utils import (
    fit_sphere as _fit_sphere,
    fit_plane as _fit_plane,
    fit_line as _fit_line,
    fit_circle as _fit_circle,
)


def fit_sphere(tree: _Tree, return_obj: bool = False) -> Sphere | Tuple:
    """Fit a sphere to tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    return_obj : bool, optional
        If True, return a vedo :class:`Sphere` object. If False, return
        ``(center, radius)``. By default False.

    Returns
    -------
    Sphere | tuple
        When *return_obj* is True, a vedo Sphere. Otherwise
        ``(center, radius)`` where *center* has shape ``(3,)``.
    """
    return _fit_sphere(vertex_coordinates(tree.graph), return_obj=return_obj)


def fit_plane(tree: _Tree, return_obj: bool = False) -> Plane | Tuple:
    """Fit a plane to tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    return_obj : bool, optional
        If True, return a vedo :class:`Plane` object. If False, return
        ``(normal, center, variances)``. By default False.

    Returns
    -------
    Plane | tuple
        When *return_obj* is True, a vedo Plane. Otherwise
        ``(normal, center, variances)``.
    """
    return _fit_plane(vertex_coordinates(tree.graph), return_obj=return_obj)


def fit_line(tree: _Tree, return_obj: bool = False) -> Line | Tuple:
    """Fit a 3D line to tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    return_obj : bool, optional
        If True, return a vedo :class:`Line` object. If False, return
        ``(slope, center, variances)``. By default False.

    Returns
    -------
    Line | tuple
        When *return_obj* is True, a vedo Line. Otherwise
        ``(slope, center, variances)``.
    """
    return _fit_line(vertex_coordinates(tree.graph), return_obj=return_obj)


def fit_circle(tree: _Tree) -> Tuple:
    """Fit a circle to tree node coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    tuple
        (center, radius, normal_to_circle).
    """
    return _fit_circle(vertex_coordinates(tree.graph))
