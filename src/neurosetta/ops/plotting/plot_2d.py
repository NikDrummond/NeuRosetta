"""2D plotting of neuron morphologies."""

from __future__ import annotations

from matplotlib.collections import LineCollection
from matplotlib.pyplot import Axes, subplots
from numpy import array, ndarray, ones, stack, vstack

from ...core import _Tree
from .synapse_plot_utils import (
    align_points_like_tree,
    categorical_rgb,
    resolve_synapse_overlay,
    type_mask,
)


def plot_2d(
    tree: _Tree,
    center: ndarray | None = None,
    ax_pad: float = 1,
    show_root: bool = True,
    line_kwargs: dict | None = None,
    point_kwargs: dict | None = None,
    root_kwargs: dict | None = None,
    axes: Axes | None = None,
    force_perspective: bool = True,
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
) -> Axes:
    """Generate a simple 2D plot of a neuron morphology.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    center : ndarray, optional
        Coordinate to center the plot at by subtracting from coordinates.
        By default np.array([0, 0]) which does nothing.
    ax_pad : float, optional
        Padding to add to the axis limits, adding a border region to the plot.
        By default 1.
    show_root : bool, optional
        Determines if to separately plot the neuron root.
        By default True.
    line_kwargs : dict, optional
        Keyword arguments passed to matplotlib.collections.LineCollection.
        By default {"color": "gray", "linewidth": 1, "alpha": 1}.
    point_kwargs : dict, optional
        Keyword arguments passed to matplotlib.pyplot.scatter for neuron nodes
        (excluding the root). By default {"color": "k"}.
    root_kwargs : dict, optional
        Keyword arguments passed to matplotlib.pyplot.scatter for the root node.
        By default {"color": "r"}. Does nothing if show_root=False.
    axes : matplotlib.pyplot.Axes, optional
        If provided, plot on these axes. Otherwise, create new Figure and Axes.
        By default None.
    force_perspective: bool, optional
        If True, the neuron is plotted aligned to maximise viewing perspective
        and orientation with the x/y axis. Synapse markers use the same frame.
        By default True.
    synapses, show_synapses : {None, bool, \"pre\", \"post\", \"both\"}, optional
        Overlay attached synapses. ``True`` / ``\"both\"`` show all; ``\"pre\"`` /
        ``\"post\"`` (and input/output aliases) filter by type. ``show_synapses``
        is an alias of ``synapses``. By default None (no overlay).
    synapse_position : {\"raw\", \"mapped\"}, optional
        Plot raw observation coordinates or mapped nearest points.
    show_synapse_mapping : bool, optional
        Draw QC segments from raw → mapped coordinates.
    pre_kwargs, post_kwargs : dict, optional
        Matplotlib scatter kwargs for pre/post synapses (ignored colour when
        *synapse_colour_by* is set).
    synapse_kwargs : dict, optional
        Shared scatter kwargs when *synapse_colour_by* is set (size/alpha etc.).
    synapse_colour_by : str, optional
        Synapse table column to colour categorically (e.g. ``partner_type``).
    synapse_cmap : str, optional
        Matplotlib colormap name for categorical colouring. By default ``tab10``.
    mapping_line_kwargs : dict, optional
        Matplotlib plot kwargs for mapping QC lines.

    Returns
    -------
    Axes
        The matplotlib Axes object with the plot.
    """
    if root_kwargs is None:
        root_kwargs = {"color": "r"}
    if point_kwargs is None:
        point_kwargs = {"color": "k"}
    if line_kwargs is None:
        line_kwargs = {"color": "gray", "linewidth": 1, "alpha": 1}
    if pre_kwargs is None:
        pre_kwargs = {"c": "#d62728", "s": 18, "alpha": 0.85, "zorder": 50}
    if post_kwargs is None:
        post_kwargs = {"c": "#1f77b4", "s": 18, "alpha": 0.85, "zorder": 50}
    if mapping_line_kwargs is None:
        mapping_line_kwargs = {"color": "0.4", "linewidth": 0.6, "alpha": 0.7, "zorder": 40}
    if center is None:
        center = array([0, 0])

    if force_perspective:
        edges = tree.get_edge_indices()
        coords = tree.align_coordinates(bind=False, robust=False)
        starts = coords[edges[:, 0]]
        stops = coords[edges[:, 1]]
    else:
        coords = tree.get_node_coordinates()
        starts, stops = tree.get_edge_coordinates()

    starts = starts[:, [0, 1]] - center
    stops = stops[:, [0, 1]] - center
    node_coords = coords[:, [0, 1]] - center

    segments = stack([starts, stops], axis=1)
    lc = LineCollection(segments, **line_kwargs)

    if axes is None:
        _, axes = subplots()

    axes.add_collection(lc)

    if show_root:
        mask = ones(node_coords.shape[0], dtype=bool)
        mask[tree.get_root_index()] = False
        axes.scatter(node_coords[~mask, 0], node_coords[~mask, 1], **root_kwargs, zorder=200)
    else:
        axes.scatter(node_coords[:, 0], node_coords[:, 1], **point_kwargs)

    mode = resolve_synapse_overlay(synapses=synapses, show_synapses=show_synapses)
    if mode is not None or show_synapse_mapping:
        _plot_synapses_2d(
            tree,
            axes,
            synapses=mode or "both",
            synapse_position=synapse_position,
            show_synapse_mapping=show_synapse_mapping,
            force_perspective=force_perspective,
            center=center,
            pre_kwargs=pre_kwargs,
            post_kwargs=post_kwargs,
            synapse_kwargs=synapse_kwargs,
            synapse_colour_by=synapse_colour_by,
            synapse_cmap=synapse_cmap,
            mapping_line_kwargs=mapping_line_kwargs,
        )

    all_pts = vstack((starts, stops))
    axes.set_xlim(all_pts[:, 0].min() - ax_pad, all_pts[:, 0].max() + ax_pad)
    axes.set_ylim(all_pts[:, 1].min() - ax_pad, all_pts[:, 1].max() + ax_pad)

    axes.set_aspect("equal")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    return axes


def _plot_synapses_2d(
    tree: _Tree,
    axes: Axes,
    *,
    synapses: str,
    synapse_position: str,
    show_synapse_mapping: bool,
    force_perspective: bool,
    center: ndarray,
    pre_kwargs: dict,
    post_kwargs: dict,
    synapse_kwargs: dict | None,
    synapse_colour_by: str | None,
    synapse_cmap: str,
    mapping_line_kwargs: dict,
) -> None:
    syn = tree.synapses
    if syn is None or len(syn) == 0:
        return

    def _frame(pts: ndarray) -> ndarray:
        if force_perspective:
            pts = align_points_like_tree(tree, pts, robust=False)
        return pts[:, [0, 1]] - center

    raw = _frame(syn.coordinates)
    mapped = None
    if syn.is_mapped:
        mapped = _frame(syn.mapped_coordinates)

    if show_synapse_mapping:
        if mapped is None:
            raise ValueError("show_synapse_mapping requires map_synapses()")
        for a, b in zip(raw, mapped, strict=True):
            axes.plot([a[0], b[0]], [a[1], b[1]], **mapping_line_kwargs)

    if synapse_position == "mapped":
        if mapped is None:
            raise ValueError("synapse_position='mapped' requires map_synapses()")
        pts = mapped
    elif synapse_position == "raw":
        pts = raw
    else:
        raise ValueError("synapse_position must be 'raw' or 'mapped'")

    df = syn.to_dataframe(copy=False)
    types = df["type"].to_numpy()
    sel = type_mask(types, synapses)  # type: ignore[arg-type]
    if not sel.any():
        return

    if synapse_colour_by is not None:
        if synapse_colour_by not in df.columns:
            raise KeyError(f"Unknown synapse column {synapse_colour_by!r}")
        style = {"s": 18, "alpha": 0.85, "zorder": 50}
        if synapse_kwargs:
            style = {**style, **synapse_kwargs}
        for key in ("c", "color", "colour"):
            style.pop(key, None)
        colours, _ = categorical_rgb(df.loc[sel, synapse_colour_by].to_numpy(), cmap=synapse_cmap)
        axes.scatter(pts[sel, 0], pts[sel, 1], c=colours, **style)
        return

    if synapses in ("pre", "both"):
        m = sel & (types == "pre")
        if m.any():
            axes.scatter(pts[m, 0], pts[m, 1], **pre_kwargs)
    if synapses in ("post", "both"):
        m = sel & (types == "post")
        if m.any():
            axes.scatter(pts[m, 0], pts[m, 1], **post_kwargs)
