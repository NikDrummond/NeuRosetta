"""3D plotting of neuron morphologies."""

from ...core import _Tree
from .viewer import Viewer


def plot_3d(
    tree: _Tree,
    show_root: bool = True,
    cache: bool = False,
    line_kwargs: dict = {"c": "k4", "lw": 1, "alpha": 1.0},
    root_kwargs: dict = {"r": 12, "c": "k4", "alpha": 1.0},
    plot_kwargs: dict = {},
    force_refresh: bool = False,
) -> Viewer:
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
    Viewer
        The vedo Viewer instance.
    """
    # set up viewer
    view = Viewer()
    view.add_neuron(
        tree,
        show_root=show_root,
        cache=cache,
        line_kwargs=line_kwargs,
        root_kwargs=root_kwargs,
        force_refresh = force_refresh
    )
    # show and set to close upon close
    view.show(**plot_kwargs).close()

    return view