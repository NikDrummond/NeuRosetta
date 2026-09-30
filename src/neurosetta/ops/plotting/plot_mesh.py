"""3D plotting helpers for mesh Stones."""

from __future__ import annotations

from typing import Any

from ...core import _Forest, _Mesh
from ...utils.vedo_utils.actors import set_actor_alpha, set_actor_colour
from .viewer import Viewer


def make_mesh_actor(
    mesh: _Mesh,
    *,
    c: Any = None,
    alpha: float | None = None,
    wireframe: bool = False,
    **kwargs: Any,
):
    """Return a styled **clone** of ``mesh.mesh`` for plotting.

    The stored :class:`~neurosetta.core.mesh._Mesh` geometry is not mutated.

    Parameters
    ----------
    mesh : _Mesh
        Neuron or neuropil mesh.
    c : Any, optional
        Vedo colour specifier.
    alpha : float, optional
        Opacity in ``[0, 1]``.
    wireframe : bool, optional
        Draw as wireframe. By default False.
    **kwargs
        Extra attributes applied via ``actor.property`` / common vedo setters
        when present (``lw``, ``lighting``, …). Unknown keys are ignored
        unless the actor exposes a matching callable/attribute.

    Returns
    -------
    vedo.Mesh
        Cloned, styled actor.
    """
    actor = mesh.mesh.clone() if hasattr(mesh.mesh, "clone") else mesh.mesh
    set_actor_colour(actor, c)
    set_actor_alpha(actor, alpha)
    if wireframe and hasattr(actor, "wireframe"):
        actor.wireframe(True)
    for key, value in kwargs.items():
        if value is None:
            continue
        attr = getattr(actor, key, None)
        if callable(attr):
            attr(value)
        elif hasattr(actor, key):
            setattr(actor, key, value)
    return actor


def plot_mesh(
    mesh: _Mesh | _Forest,
    *,
    c: Any = None,
    alpha: float | None = 0.5,
    wireframe: bool = False,
    colour_cycle: bool = True,
    plot_kwargs: dict | None = None,
    return_viewer: bool = False,
    **style_kwargs: Any,
) -> Viewer | Any:
    """Show a 3D plot of one mesh or a mesh collection.

    For an attached neuron facet, prefer
    ``tree.show_3d(show_mesh=True)`` so skeleton and surface share one viewer.

    Parameters
    ----------
    mesh : Tree_mesh, Neuropil, Forest_mesh, or Neuropils
        Mesh object or collection.
    c : Any, optional
        Colour for a single mesh. For collections, ignored when
        *colour_cycle* is True.
    alpha : float, optional
        Opacity. By default 0.5.
    wireframe : bool, optional
        Wireframe mode. By default False.
    colour_cycle : bool, optional
        When plotting a collection, assign a distinct colour per member.
        By default True.
    plot_kwargs : dict, optional
        Forwarded to :meth:`~neurosetta.ops.plotting.viewer.Viewer.show`.
    return_viewer : bool, optional
        If True, return the :class:`~neurosetta.ops.plotting.viewer.Viewer`
        without calling ``show``. By default False.
    **style_kwargs
        Extra style kwargs forwarded to :func:`make_mesh_actor`.

    Returns
    -------
    Viewer | Any
        Viewer when *return_viewer* is True; otherwise the result of
        ``Viewer.show`` (desktop plotter or notebook widget).
    """
    viewer = Viewer()
    viewer.add_mesh(
        mesh,
        c=c,
        alpha=alpha,
        wireframe=wireframe,
        colour_cycle=colour_cycle,
        **style_kwargs,
    )
    if return_viewer:
        return viewer
    return viewer.show(**(plot_kwargs or {}))
