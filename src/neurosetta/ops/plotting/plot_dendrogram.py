"""Dendrogram plotting for neuron trees."""

from graph_tool import VertexPropertyMap
from matplotlib.collections import LineCollection
from matplotlib.pyplot import Axes, subplots
from numpy import arange, stack, vstack

from ...core import _Tree
from ...utils.graph_utils import PostOrderVisitor
from ..tree_graphs import get_max_width, get_node_depth, get_post_order


def compute_dend_x(
    tree: _Tree,
    post_order: PostOrderVisitor,
) -> dict[int, float]:
    """Compute relative x coordinates from the tree topology.

    Leaves are assigned consecutive x positions. Internal vertices are
    positioned at the mean x-coordinate of their children.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    post_order : PostOrderVisitor
        Post-order traversal of the tree.

    Returns
    -------
    dict[int, float]
        Mapping from vertex index to relative x-coordinate.
    """
    x: dict[int, float] = {}
    leaf_index = 0

    for vertex in post_order.post_order:
        vertex = int(vertex)
        children = tree.graph.get_out_neighbors(vertex)

        if children.size == 0:
            x[vertex] = float(leaf_index)
            leaf_index += 1
        else:
            x[vertex] = sum(x[int(child)] for child in children) / children.size

    return x


def plot_dendrogram(
    tree: _Tree,
    x_width: float | None = None,
    x_spacing: float | None = None,
    x_pad: float = 1.0,
    y_pad: float = 2.0,
    root_position: str = "bottom",
    # ylabel: bool = False,
    line_kwargs: dict | None = None,
    axes: Axes | None = None,
) -> Axes:
    """Plot a neuron tree as a dendrogram.

    Horizontal positions are determined from the tree topology. Leaves are
    assigned consecutive positions and internal vertices are positioned at
    the mean x-coordinate of their children. The resulting coordinates can
    either be scaled to a specified total width or use a fixed spacing
    between consecutive leaves.

    Parameters
    ----------
    tree : _Tree
        Neuron tree to plot.

    x_width : float or None, optional
        Desired horizontal width of the dendrogram. If ``None``, the maximum
        width returned by :func:`get_max_width` is used. Ignored when
        ``x_spacing`` is provided.

    x_spacing : float or None, optional
        Spacing between consecutive leaf positions. If provided, this takes
        precedence over ``x_width``. By default, ``None``.

    x_pad : float, optional
        Horizontal padding added to the x-axis limits on either side of the
        dendrogram. By default, 1.0.

    y_pad : float, optional
        Vertical padding added to the y-axis limits on either side of the
        dendrogram. By default, 1.0.

    root_position : {"bottom", "top"}, optional
        Position of the root on the plot. ``"bottom"`` places the root at
        the bottom of the dendrogram, while ``"top"`` inverts the y-axis so
        that the root appears at the top. By default, ``"bottom"``.

    ylabel : bool, optional
        Whether to display the y-axis label. By default, ``False``.

    line_kwargs : dict or None, optional
        Keyword arguments passed to
        :class:`matplotlib.collections.LineCollection` when drawing the
        dendrogram edges. This can be used to customize properties such as
        line color, linewidth, linestyle, alpha, and cap style.

        For example::

            line_kwargs={
                "color": "black",
                "linewidth": 2,
                "alpha": 0.8,
            }

        If ``None``, the default line properties are used.

    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes on which to draw the dendrogram. If ``None``, a new
        figure and axes are created. By default, ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object containing the dendrogram.

    Raises
    ------
    ValueError
        If ``root_position`` is not ``"top"`` or ``"bottom"``.
    ValueError
        If ``x_width`` is provided and is not positive.
    ValueError
        If ``x_spacing`` is provided and is not positive.

    Notes
    -----
    The root vertex is assumed to have index 0.

    The graph is assumed to be directed, with outgoing edges representing
    parent-to-child relationships.

    ``x_width`` and ``x_spacing`` describe different aspects of the
    horizontal layout. ``x_width`` controls the total horizontal extent of
    the dendrogram, whereas ``x_spacing`` controls the distance between
    consecutive leaves. If both are provided, ``x_spacing`` takes
    precedence.

    The default horizontal width is determined by ``get_max_width(tree)``.
    This controls the overall scale of the dendrogram but does not determine
    the number of leaf positions, which is determined by the tree topology.
    """
    if root_position not in {"top", "bottom"}:
        raise ValueError(
            f"root_position must be 'top' or 'bottom', got {root_position!r}"
        )

    if x_width is not None and x_width <= 0:
        raise ValueError(f"x_width must be positive, got {x_width}")

    if x_spacing is not None and x_spacing <= 0:
        raise ValueError(f"x_spacing must be positive, got {x_spacing}")

    if line_kwargs is None:
        line_kwargs = {}

    depth = get_node_depth(tree, bind=False)
    post_order = get_post_order(tree, bind=False)
    x = compute_dend_x(tree, post_order)

    # Convert topology-based x coordinates into physical coordinates.
    #
    # x_spacing takes precedence over x_width. Otherwise, x_width defaults
    # to the maximum width of the tree.
    if x_spacing is not None:
        x = {vertex: value * x_spacing for vertex, value in x.items()}
    else:
        if x_width is None:
            x_width = float(get_max_width(tree))

        x_min = min(x.values())
        x_max = max(x.values())
        topology_width = x_max - x_min

        if topology_width > 0:
            scale = x_width / topology_width
            x = {vertex: (value - x_min) * scale for vertex, value in x.items()}
        else:
            # Single-node tree.
            x = {vertex: 0.0 for vertex in x}

    if axes is None:
        _, axes = subplots()

    # Build a coordinate array indexed by graph vertex index.
    n_vertices = tree.graph.num_vertices()
    coords = stack(
        (
            [x[i] for i in range(n_vertices)],
            [depth[i] for i in range(n_vertices)],
        ),
        axis=1,
    )

    edges = tree.get_edge_indices()

    if edges.size == 0:
        # Handle a tree consisting only of the root.
        root_x = coords[0, 0]
        root_y = coords[0, 1]

        axes.set_xlim(
            root_x - x_pad,
            root_x + x_pad,
        )
        axes.set_ylim(
            root_y - y_pad,
            root_y + y_pad,
        )
    else:
        starts = coords[edges[:, 0]]
        stops = coords[edges[:, 1]]

        segments = stack(
            (starts, stops),
            axis=1,
        )

        line_collection = LineCollection(
            segments,
            **line_kwargs,
        )
        axes.add_collection(line_collection)

        all_points = vstack(
            (starts, stops),
        )

        axes.set_xlim(
            all_points[:, 0].min() - x_pad,
            all_points[:, 0].max() + x_pad,
        )
        axes.set_ylim(
            all_points[:, 1].min() - y_pad,
            all_points[:, 1].max() + y_pad,
        )

    if root_position == "top":
        axes.yaxis.set_inverted(True)

    # Highlight the root vertex. The root is guaranteed to be vertex 0.
    axes.scatter(
        coords[0, 0],
        coords[0, 1],
        c="r",
        zorder=100,
    )

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)

    axes.set_xticks([])

    # if ylabel:
    #     axes.set_ylabel("Tree Depth")
    # else:
    #     # axes.spines["left"].set_visible(False)
    axes.set_ylabel("Tree Depth")
    axes.set_aspect("auto")

    return axes
