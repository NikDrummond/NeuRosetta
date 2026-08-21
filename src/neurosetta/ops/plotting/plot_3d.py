"""3D plotting of neuron morphologies."""

from __future__ import annotations

from typing import Any

from ...core import _Tree
from .viewer import Viewer


def plot_3d(
    tree: _Tree,
    show_root: bool | None = None,
    cache: bool = True,
    line_kwargs: dict | None = None,
    root_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
    force_refresh: bool = False,
    **style_kwargs,
) -> Viewer | Any:
    """Show a 3D plot of *tree* using its :attr:`~neurosetta.core.tree._Tree.plot3d` handle.

    Style already stored on ``tree.plot3d`` is honoured. Arguments here are optional
    overrides for this call; when *cache* is True they are written back onto the
    tree's plot.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    show_root : bool | None, optional
        Whether to show the root (soma) marker. None leaves the plot's current
        setting (True on a fresh tree). By default None.
    cache : bool, optional
        When True, build and display ``tree.plot3d``. When False, display a
        standalone copy that inherits the stored style without mutating the
        tree's plot. By default True.
    line_kwargs : dict | None, optional
        Extra vedo ``Lines`` kwargs. Recognised style keys (``c``/``lw``/``alpha``)
        override stored style; anything else is forwarded to the constructor.
        By default None.
    root_kwargs : dict | None, optional
        Extra root-marker kwargs, handled as *line_kwargs*. By default None.
    plot_kwargs : dict | None, optional
        Keyword arguments forwarded to :meth:`~neurosetta.ops.plotting.viewer.Viewer.show`.
    force_refresh : bool, optional
        Rebuild vedo actors even if they already exist. By default False.
    **style_kwargs
        Further style overrides accepted by
        :meth:`~neurosetta.ops.plotting.utils.TreePlot3D.set_style`.

    Returns
    -------
    Viewer | Any
        The :class:`~neurosetta.ops.plotting.viewer.Viewer` for desktop ``vtk``
        backends. For notebook backends (e.g. ``k3d``), the inline widget
        returned by vedo ``show()``.
    """
    plot = tree.make_plot3d(
        show_root=show_root,
        line_kwargs=line_kwargs,
        root_kwargs=root_kwargs,
        cache=cache,
        force_refresh=force_refresh,
        **style_kwargs,
    )
    return plot.show(**(plot_kwargs or {}))
