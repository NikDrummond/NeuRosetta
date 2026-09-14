"""3D plotting of neuron morphologies."""

from __future__ import annotations

from typing import Any

from ...core import _Tree
from .synapse_plot_utils import resolve_synapse_overlay
from .viewer import Viewer


def plot_3d(
    tree: _Tree,
    show_root: bool | None = None,
    cache: bool = True,
    line_kwargs: dict | None = None,
    root_kwargs: dict | None = None,
    plot_kwargs: dict | None = None,
    force_refresh: bool = False,
    synapses: bool | str | None = None,
    show_synapses: bool | str | None = None,
    synapse_position: str = "raw",
    show_synapse_mapping: bool = False,
    pre_kwargs: dict | None = None,
    post_kwargs: dict | None = None,
    synapse_kwargs: dict | None = None,
    synapse_colour_by: str | None = None,
    synapse_cmap: str = "tab10",
    mapping_line_kwargs: dict | None = None,
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
    synapses, show_synapses : {None, bool, \"pre\", \"post\", \"both\"}, optional
        Overlay attached synapses as point clouds. ``True`` means all types;
        ``show_synapses`` aliases ``synapses``. By default None.
    synapse_position : {\"raw\", \"mapped\"}, optional
        Plot raw or mapped synapse coordinates.
    show_synapse_mapping : bool, optional
        Draw QC lines from raw → mapped locations.
    pre_kwargs, post_kwargs : dict, optional
        Vedo Points kwargs for pre/post synapses.
    synapse_kwargs : dict, optional
        Shared Points kwargs when *synapse_colour_by* is set.
    synapse_colour_by : str, optional
        Synapse table column for categorical colouring.
    synapse_cmap : str, optional
        Matplotlib colormap name for categorical colours. By default ``tab10``.
    mapping_line_kwargs : dict, optional
        Vedo Lines kwargs for mapping QC segments.
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
    mode = resolve_synapse_overlay(synapses=synapses, show_synapses=show_synapses)
    if mode is None and not show_synapse_mapping:
        return plot.show(**(plot_kwargs or {}))

    viewer = Viewer()
    viewer.add(*plot.actors)
    viewer.add_synapses(
        tree,
        synapses=mode or "both",
        synapse_position=synapse_position,
        show_synapse_mapping=show_synapse_mapping,
        pre_kwargs=pre_kwargs,
        post_kwargs=post_kwargs,
        synapse_kwargs=synapse_kwargs,
        synapse_colour_by=synapse_colour_by,
        synapse_cmap=synapse_cmap,
        mapping_line_kwargs=mapping_line_kwargs,
    )
    return viewer.show(**(plot_kwargs or {}))
