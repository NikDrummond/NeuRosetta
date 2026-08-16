"""3D plotting of neuron morphologies."""

from typing import Any

from ...config.vedo_settings import is_notebook_vedo_backend
from ...core import _Tree
from .viewer import Viewer


def plot_3d(
    tree: _Tree,
    show_root: bool = True,
    cache: bool = False,
    line_kwargs: dict = None,
    root_kwargs: dict = None,
    plot_kwargs: dict = None,
    force_refresh: bool = False,
) -> Viewer | Any:
    """On-the-fly 3D neuron plotting. Opens an interactive vedo.Plotter instance.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    show_root : bool, optional
        Whether to show the root (soma) marker. By default True.
    cache : bool, optional
        Whether to keep the generated plotting object. Otherwise it is generated
        each time. By default False.
    line_kwargs : dict, optional
        Keyword arguments passed to vedo.Lines to customise the neuron appearance.
        By default {"c": "k4", "lw": 1, "alpha": 1.0}.
    root_kwargs : dict, optional
        Keyword arguments passed to vedo.Point to customise the root marker.
        By default {"r": 12, "c": "k4", "alpha": 1.0}.
    plot_kwargs : dict, optional
        Keyword arguments passed to vedo.show.
    force_refresh : bool, optional
        Force rebuild of the plot objects even if cached. By default False.

    Returns
    -------
    Viewer | Any
        The :class:`~neurosetta.ops.plotting.viewer.Viewer` for desktop ``vtk``
        backends. For notebook backends (e.g. ``k3d``), the inline widget
        returned by vedo ``show()``.
    """
    # set up viewer
    if plot_kwargs is None:
        plot_kwargs = {}
    if root_kwargs is None:
        root_kwargs = {"r": 12, "c": "k4", "alpha": 1.0}
    if line_kwargs is None:
        line_kwargs = {"c": "k4", "lw": 1, "alpha": 1.0}
    view = Viewer()
    view.add_neuron(
        tree,
        show_root=show_root,
        cache=cache,
        line_kwargs=line_kwargs,
        root_kwargs=root_kwargs,
        force_refresh=force_refresh,
    )
    shown = view.show(**plot_kwargs)
    if is_notebook_vedo_backend():
        return shown
    shown.close()
    return view
