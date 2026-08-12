"""2D plotting of neuron morphologies."""

from numpy import array, stack, vstack, ones, ndarray
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Axes, subplots

from ...core import _Tree


def plot_2d(
    tree: _Tree,
    center: ndarray = array([0, 0]),
    ax_pad: float = 1,
    show_root: bool = True,
    line_kwargs: dict = {"color": "gray", "linewidth": 1, "alpha": 1},
    point_kwargs: dict = {"color": "k"},
    root_kwargs: dict = {"color": "r"},
    axes: Axes | None = None,
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

    Returns
    -------
    Axes
        The matplotlib Axes object with the plot.
    """
    # get source and target coordinates
    starts, stops = tree.get_edge_coordinates()
    # remove z axis
    starts = starts[:, [0, 1]]
    stops = stops[:, [0, 1]]

    starts -= center
    stops -= center
    # stack starts and end into a segments array
    segments = stack([starts, stops], axis=1)
    # create LineCollection
    lc = LineCollection(segments, **line_kwargs)

    if axes is None:
        fig, axes = subplots()

    axes.add_collection(lc)

    # plot points
    coords = tree.get_node_coordinates()

    if show_root:
        # generate mask to subset out root
        mask = ones(coords.shape[0], dtype=bool)
        mask[tree.get_root_index()] = False
        axes.scatter(coords[mask, 0], coords[mask, 1], **point_kwargs)
        axes.scatter(coords[~mask, 0], coords[~mask, 1], **root_kwargs)
    else:
        axes.scatter(coords[:, 0], coords[:, 1], **point_kwargs)

    # adjust limits
    all_pts = vstack((starts, stops))
    axes.set_xlim(all_pts[:, 0].min() - ax_pad, all_pts[:, 0].max() + ax_pad)
    axes.set_ylim(all_pts[:, 1].min() - ax_pad, all_pts[:, 1].max() + ax_pad)

    axes.set_aspect("equal")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    return axes