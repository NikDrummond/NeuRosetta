"""Dendrogram plotting for neuron trees."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from matplotlib.collections import LineCollection
from matplotlib.pyplot import Axes, subplots
from numpy import (
    arange,
    array,
    concatenate,
    cos,
    empty,
    linspace,
    ndarray,
    pi,
    repeat,
    sin,
    stack,
    vstack,
    zeros_like,
)

from ...core import _Tree
from ...utils.graph_utils import PostOrderVisitor
from ..tree_graphs import get_max_width, get_node_depth, get_post_order, reduce_tree

Orientation = Literal["vertical", "horizontal", "radial"]
EdgeStyle = Literal["diagonal", "orthogonal"]
RootPosition = Literal["top", "bottom"]


@contextmanager
def _temporary_tree_reduction(tree: _Tree) -> Iterator[None]:
    """Temporarily reduce ``tree`` for plotting, restoring on exit."""
    original_graph = None
    if not tree.is_reduced():
        original_graph = tree.graph
        reduce_tree(tree, inplace=True)
    try:
        yield
    finally:
        if original_graph is not None:
            tree.graph = original_graph


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


def _compute_leaf_mask(tree: _Tree, n_vertices: int) -> ndarray:
    """Compute a boolean mask marking which vertices are leaves.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    n_vertices : int
        Number of vertices in the tree.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(n_vertices,)``, ``True`` for leaves.
    """
    return array(
        [tree.graph.get_out_neighbors(v).size == 0 for v in range(n_vertices)],
        dtype=bool,
    )


def _compute_leaf_extension_segments(
    branch: ndarray,
    depth: ndarray,
    aligned_depth: ndarray,
    leaf_mask: ndarray,
    orientation: Orientation,
) -> ndarray:
    """Build straight guide segments extending leaves to their aligned depth.

    Each segment runs from a leaf's true position straight out to its
    aligned position, at a fixed branch position (or angle, for
    ``orientation="radial"``). Because branch position never changes along
    a segment, these guides are always straight lines and never cross
    unrelated parts of the tree, regardless of ``edge_style``.

    Parameters
    ----------
    branch : numpy.ndarray
        Topology-based branch position of every vertex, shape ``(n,)``.
        Used in full (not just leaves) so the radial angle normalization
        matches the rest of the plot.
    depth : numpy.ndarray
        True tree depth of every vertex, shape ``(n,)``.
    aligned_depth : numpy.ndarray
        Depth each vertex should appear at once aligned, shape ``(n,)``.
        Only the leaf entries are used.
    leaf_mask : numpy.ndarray
        Boolean array of shape ``(n,)``, ``True`` for leaves.
    orientation : Orientation
        Selected dendrogram orientation.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_segments, 2, 2)`` suitable for
        :class:`matplotlib.collections.LineCollection`. Leaves already at
        the aligned depth are skipped, so this may have fewer than
        ``leaf_mask.sum()`` segments.
    """
    leaf_branch = branch[leaf_mask]
    leaf_depth = depth[leaf_mask]
    leaf_aligned_depth = aligned_depth[leaf_mask]

    # Skip leaves that are already at the aligned depth; they need no
    # guide line, which also keeps clutter down.
    needs_extension = leaf_depth < leaf_aligned_depth
    leaf_branch = leaf_branch[needs_extension]
    leaf_depth = leaf_depth[needs_extension]
    leaf_aligned_depth = leaf_aligned_depth[needs_extension]

    if leaf_branch.size == 0:
        return empty((0, 2, 2))

    if orientation == "vertical":
        starts = stack((leaf_branch, leaf_depth), axis=1)
        stops = stack((leaf_branch, leaf_aligned_depth), axis=1)
    elif orientation == "horizontal":
        starts = stack((leaf_depth, leaf_branch), axis=1)
        stops = stack((leaf_aligned_depth, leaf_branch), axis=1)
    elif orientation == "radial":
        branch_min = branch.min()
        branch_range = branch.max() - branch_min

        start_x, start_y = _polar_transform(leaf_branch, leaf_depth, branch_min, branch_range)
        stop_x, stop_y = _polar_transform(leaf_branch, leaf_aligned_depth, branch_min, branch_range)
        starts = stack((start_x, start_y), axis=1)
        stops = stack((stop_x, stop_y), axis=1)
    else:
        raise ValueError(
            f"Unknown orientation {orientation!r}. Expected 'vertical', 'horizontal', or 'radial'."
        )

    return stack((starts, stops), axis=1)


def _polar_transform(
    branch: ndarray,
    depth: ndarray,
    branch_min: float,
    branch_range: float,
) -> tuple[ndarray, ndarray]:
    """Map (branch position, depth) pairs to (x, y) points on a circle.

    ``depth`` becomes the radial distance from the origin and ``branch``
    (normalized against ``branch_min``/``branch_range``) becomes the angle.
    """
    if branch_range > 0:
        theta = 2.0 * pi * (branch - branch_min) / branch_range
    else:
        theta = zeros_like(branch, dtype=float)

    x = depth * cos(theta)
    y = depth * sin(theta)

    return x, y


def _compute_vertex_coords(
    branch: ndarray,
    depth: ndarray,
    orientation: Orientation,
) -> ndarray:
    """Compute the final plotted (x, y) coordinate of every vertex.

    Parameters
    ----------
    branch : numpy.ndarray
        Topology-based branch position of each vertex, shape ``(n,)``.
    depth : numpy.ndarray
        Tree depth of each vertex, shape ``(n,)``.
    orientation : Orientation
        Selected dendrogram orientation.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n, 2)`` with the final plotted coordinates.
    """
    if orientation == "vertical":
        return stack((branch, depth), axis=1)

    if orientation == "horizontal":
        return stack((depth, branch), axis=1)

    if orientation == "radial":
        branch_min = branch.min()
        branch_range = branch.max() - branch_min
        x, y = _polar_transform(branch, depth, branch_min, branch_range)
        return stack((x, y), axis=1)

    raise ValueError(
        f"Unknown orientation {orientation!r}. Expected 'vertical', 'horizontal', or 'radial'."
    )


def _compute_segments(
    edges: ndarray,
    branch: ndarray,
    depth: ndarray,
    orientation: Orientation,
    edge_style: EdgeStyle,
    arc_resolution: int,
) -> ndarray:
    """Build line segments for the selected orientation and edge style.

    Parameters
    ----------
    edges : numpy.ndarray
        Array of shape ``(n_edges, 2)`` giving parent/child vertex index
        pairs, as returned by ``tree.get_edge_indices()``.
    branch : numpy.ndarray
        Topology-based branch position of each vertex, shape ``(n,)``.
    depth : numpy.ndarray
        Tree depth of each vertex, shape ``(n,)``.
    orientation : Orientation
        Selected dendrogram orientation.
    edge_style : EdgeStyle
        ``"diagonal"`` draws a single straight line per edge.
        ``"orthogonal"`` draws a right-angle "elbow" per edge: a move along
        depth at the parent's branch position, followed by a move along
        branch position at the child's depth. For ``orientation="radial"``,
        this becomes a straight radial line (parent depth to child depth,
        at the parent's angle) followed by an arc at the child's depth,
        sampled into ``arc_resolution`` points.
    arc_resolution : int
        Number of points used to approximate each arc when
        ``orientation="radial"`` and ``edge_style="orthogonal"``. Ignored
        otherwise.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_segments, 2, 2)`` suitable for
        :class:`matplotlib.collections.LineCollection`.
    """
    if edges.shape[0] == 0:
        return empty((0, 2, 2))

    branch_parent = branch[edges[:, 0]]
    depth_parent = depth[edges[:, 0]]
    branch_child = branch[edges[:, 1]]
    depth_child = depth[edges[:, 1]]

    if orientation == "radial":
        branch_min = branch.min()
        branch_range = branch.max() - branch_min

        if edge_style == "diagonal":
            start_x, start_y = _polar_transform(
                branch_parent, depth_parent, branch_min, branch_range
            )
            stop_x, stop_y = _polar_transform(branch_child, depth_child, branch_min, branch_range)
            starts = stack((start_x, start_y), axis=1)
            stops = stack((stop_x, stop_y), axis=1)

            return stack((starts, stops), axis=1)

        # Orthogonal radial edge: a straight radial line from the parent's
        # depth to the child's depth (angle fixed at the parent's branch
        # position), followed by an arc at the child's depth sweeping from
        # the parent's angle to the child's angle.
        n_arc_points = max(arc_resolution, 2)

        radial_start_x, radial_start_y = _polar_transform(
            branch_parent, depth_parent, branch_min, branch_range
        )
        radial_stop_x, radial_stop_y = _polar_transform(
            branch_parent, depth_child, branch_min, branch_range
        )

        radial_segments = stack(
            (
                stack((radial_start_x, radial_start_y), axis=1),
                stack((radial_stop_x, radial_stop_y), axis=1),
            ),
            axis=1,
        )

        # Sample each arc as (n_arc_points - 1) straight sub-segments.
        branch_samples = linspace(branch_parent, branch_child, n_arc_points, axis=1)
        depth_samples = repeat(depth_child[:, None], n_arc_points, axis=1)
        arc_x, arc_y = _polar_transform(branch_samples, depth_samples, branch_min, branch_range)
        arc_points = stack((arc_x, arc_y), axis=2)
        arc_segments = stack((arc_points[:, :-1], arc_points[:, 1:]), axis=2).reshape(-1, 2, 2)

        return concatenate((radial_segments, arc_segments), axis=0)

    # Cartesian orientations (vertical / horizontal), worked out in
    # logical (branch, depth) space and swapped for "horizontal" at the
    # end, since that orientation is just (branch, depth) with the axes
    # exchanged.
    if edge_style == "diagonal":
        starts = stack((branch_parent, depth_parent), axis=1)
        stops = stack((branch_child, depth_child), axis=1)
        segments = stack((starts, stops), axis=1)
    elif edge_style == "orthogonal":
        starts = stack((branch_parent, depth_parent), axis=1)
        elbow = stack((branch_parent, depth_child), axis=1)
        stops = stack((branch_child, depth_child), axis=1)

        first_leg = stack((starts, elbow), axis=1)
        second_leg = stack((elbow, stops), axis=1)
        segments = concatenate((first_leg, second_leg), axis=0)
    else:
        raise ValueError(f"Unknown edge_style {edge_style!r}. Expected 'diagonal' or 'orthogonal'.")

    if orientation == "horizontal":
        segments = segments[..., ::-1]

    return segments


def plot_dendrogram(
    tree: _Tree,
    orientation: Orientation = "vertical",
    edge_style: EdgeStyle = "orthogonal",
    align_leaves: bool = False,
    x_width: float | None = None,
    x_spacing: float | None = None,
    x_pad: float = 1.0,
    y_pad: float = 1.0,
    root_position: RootPosition = "bottom",
    ylabel: bool = False,
    line_kwargs: dict | None = None,
    leaf_line_kwargs: dict | None = None,
    axes: Axes | None = None,
    arc_resolution: int = 32,
) -> Axes:
    """Plot a neuron tree using a dendrogram-style layout.

    Parameters
    ----------
    tree : _Tree
        Neuron tree to plot.

    orientation : {"vertical", "horizontal", "radial"}, optional
        Coordinate system used to arrange the tree.

        ``"vertical"``
            Depth is represented on the y-axis and the root is at the
            bottom by default.

        ``"horizontal"``
            Depth is represented on the x-axis and the branching order is
            represented on the y-axis.

        ``"radial"``
            Depth determines the radial distance from the origin and the
            leaf ordering determines the angular position.

        By default, ``"vertical"`` is used.

    edge_style : {"diagonal", "orthogonal"}, optional
        How parent-to-child edges are drawn.

        ``"diagonal"``
            A single straight line from parent to child.

        ``"orthogonal"``
            A right-angle "elbow": a move along depth at the parent's
            branch position, followed by a move along branch position at
            the child's depth. For ``orientation="radial"``, this becomes
            a straight radial line followed by an arc (see
            ``arc_resolution``).

        By default, ``"orthogonal"`` is used.

    align_leaves : bool, optional
        If ``True``, every leaf gets a straight guide line extending it
        out to the tree's maximum depth (or maximum radius, for
        ``orientation="radial"``), so leaf tips line up visually. The real
        tree edges are left untouched — only a short, separate segment is
        added per leaf, running from the leaf's true position straight
        out to the aligned depth at a fixed branch position. This keeps
        the guide lines from crossing other branches, unlike stretching
        the real edges would. Leaves already at the maximum depth get no
        guide line. Style the guides with ``leaf_line_kwargs``. Applies
        uniformly across every ``orientation``/``edge_style`` combination.
        By default, ``False``.

    x_width : float or None, optional
        Desired width of the dendrogram layout. If ``None``, the maximum
        width returned by :func:`get_max_width` is used. This parameter is
        ignored when ``x_spacing`` is provided.

        For ``"vertical"`` and ``"horizontal"`` orientations, this controls
        the branch-position extent. For ``"radial"``, it controls the
        angular distribution of the leaves indirectly through the topology
        layout.

    x_spacing : float or None, optional
        Spacing between consecutive leaf positions. If provided, this takes
        precedence over ``x_width``. By default, ``None``.

    x_pad : float, optional
        Padding added to the first coordinate axis on either side of the
        plotted tree. For vertical/horizontal orientations this is
        horizontal padding. For radial orientation it pads the circular
        extent in both directions. By default, 1.0.

    y_pad : float, optional
        Padding added to the second coordinate axis on either side of the
        plotted tree. For vertical/horizontal orientations this is
        vertical padding. For radial orientation it pads the circular
        extent in both directions. By default, 1.0.

    root_position : {"bottom", "top"}, optional
        Position of the root for the ``"vertical"`` and ``"horizontal"``
        orientations. ``"bottom"`` places the root at the minimum depth
        coordinate, while ``"top"`` reverses the depth axis. By default,
        ``"bottom"``.

        This parameter has no effect when ``orientation="radial"``.

    ylabel : bool, optional
        Whether to display the y-axis label. By default, ``False``.

    line_kwargs : dict or None, optional
        Keyword arguments passed directly to
        :class:`matplotlib.collections.LineCollection` when drawing tree
        edges. This can be used to customize line appearance.

        Examples
        --------
        ``line_kwargs={"color": "black", "linewidth": 2}``

        ``line_kwargs={"color": "gray", "linewidth": 0.5, "alpha": 0.5}``

        Any keyword argument accepted by ``LineCollection`` may be supplied.
        By default, ``None``.

    leaf_line_kwargs : dict or None, optional
        Keyword arguments passed to :class:`matplotlib.collections.LineCollection`
        for the leaf-alignment guide lines added when ``align_leaves=True``.
        Ignored otherwise. If ``None``, defaults to a thin dashed gray
        style (``{"color": "0.6", "linestyle": "--", "linewidth": 0.8}``)
        so guides read as distinct from real branches at a glance. By
        default, ``None``.

    axes : matplotlib.axes.Axes or None, optional
        Matplotlib axes on which to draw the dendrogram. If ``None``, a new
        figure and axes are created. By default, ``None``.

    arc_resolution : int, optional
        Number of points used to approximate each arc when
        ``orientation="radial"`` and ``edge_style="orthogonal"``. Higher
        values give smoother arcs at the cost of more line segments.
        Ignored for all other orientation/edge_style combinations. By
        default, 32.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes object containing the dendrogram.

    Raises
    ------
    ValueError
        If ``tree`` has no vertices.
    ValueError
        If ``orientation`` is not one of ``"vertical"``, ``"horizontal"``,
        or ``"radial"``.
    ValueError
        If ``edge_style`` is not one of ``"diagonal"`` or ``"orthogonal"``.
    ValueError
        If ``root_position`` is not ``"top"`` or ``"bottom"``.
    ValueError
        If ``x_width`` is provided and is not positive.
    ValueError
        If ``x_spacing`` is provided and is not positive.
    ValueError
        If ``arc_resolution`` is provided and is less than 2.

    Notes
    -----
    The root vertex is assumed to have index 0.

    The graph is assumed to be directed, with outgoing edges representing
    parent-to-child relationships.

    The tree topology determines the relative ordering of leaves. Internal
    vertices are positioned at the mean coordinate of their children.

    For radial orientation, the root is located at the center and depth
    increases with radial distance. Leaf ordering determines the angular
    position around the circle.
    """
    valid_orientations = {"vertical", "horizontal", "radial"}
    valid_edge_styles = {"diagonal", "orthogonal"}

    if orientation not in valid_orientations:
        raise ValueError(
            f"orientation must be one of {sorted(valid_orientations)}, got {orientation!r}"
        )

    if edge_style not in valid_edge_styles:
        raise ValueError(
            f"edge_style must be one of {sorted(valid_edge_styles)}, got {edge_style!r}"
        )

    if root_position not in {"top", "bottom"}:
        raise ValueError(f"root_position must be 'top' or 'bottom', got {root_position!r}")

    if x_width is not None and x_width <= 0:
        raise ValueError(f"x_width must be positive, got {x_width}")

    if x_spacing is not None and x_spacing <= 0:
        raise ValueError(f"x_spacing must be positive, got {x_spacing}")

    if arc_resolution < 2:
        raise ValueError(f"arc_resolution must be at least 2, got {arc_resolution}")

    if tree.graph.num_vertices() == 0:
        raise ValueError("Cannot plot a dendrogram for an empty tree.")

    with _temporary_tree_reduction(tree):
        return _plot_dendrogram_on_axes(
            tree,
            orientation=orientation,
            edge_style=edge_style,
            align_leaves=align_leaves,
            x_width=x_width,
            x_spacing=x_spacing,
            x_pad=x_pad,
            y_pad=y_pad,
            root_position=root_position,
            ylabel=ylabel,
            line_kwargs=line_kwargs,
            leaf_line_kwargs=leaf_line_kwargs,
            axes=axes,
            arc_resolution=arc_resolution,
        )


def _plot_dendrogram_on_axes(
    tree: _Tree,
    orientation: Orientation,
    edge_style: EdgeStyle,
    align_leaves: bool,
    x_width: float | None,
    x_spacing: float | None,
    x_pad: float,
    y_pad: float,
    root_position: RootPosition,
    ylabel: bool,
    line_kwargs: dict | None,
    leaf_line_kwargs: dict | None,
    axes: Axes | None,
    arc_resolution: int,
) -> Axes:
    if line_kwargs is None:
        line_kwargs = {}

    if leaf_line_kwargs is None:
        leaf_line_kwargs = {"color": "0.6", "linestyle": "--", "linewidth": 0.8}

    n_vertices = tree.graph.num_vertices()

    depth_map = get_node_depth(tree, bind=False)
    post_order = get_post_order(tree, bind=False)

    x = compute_dend_x(tree, post_order)
    branch = array([x[i] for i in range(n_vertices)], dtype=float)
    depth = array([float(depth_map[i]) for i in range(n_vertices)], dtype=float)

    leaf_mask = _compute_leaf_mask(tree, n_vertices) if align_leaves else None
    aligned_depth = depth
    if align_leaves:
        aligned_depth = depth.copy()
        aligned_depth[leaf_mask] = depth.max()

    # Convert topology-based branch positions into physical coordinates.
    if x_spacing is not None:
        branch = branch * x_spacing
    else:
        if x_width is None:
            x_width = float(get_max_width(tree))

        branch_min = branch.min()
        branch_max = branch.max()
        topology_width = branch_max - branch_min

        if topology_width > 0:
            scale = x_width / topology_width
            branch = (branch - branch_min) * scale
        else:
            branch = zeros_like(branch)

    if axes is None:
        _, axes = subplots()

    coords = _compute_vertex_coords(branch, depth, orientation)
    coords_bounds = (
        _compute_vertex_coords(branch, aligned_depth, orientation) if align_leaves else coords
    )
    edges = tree.get_edge_indices()

    if orientation == "radial":
        if edges.size:
            segments = _compute_segments(
                edges,
                branch,
                depth,
                orientation,
                edge_style,
                arc_resolution,
            )

            axes.add_collection(
                LineCollection(
                    segments,
                    **line_kwargs,
                )
            )

        if align_leaves:
            extension_segments = _compute_leaf_extension_segments(
                branch, depth, aligned_depth, leaf_mask, orientation
            )

            if extension_segments.shape[0]:
                axes.add_collection(
                    LineCollection(
                        extension_segments,
                        **leaf_line_kwargs,
                    )
                )

        all_x = coords_bounds[:, 0]
        all_y = coords_bounds[:, 1]

        radius = max(
            abs(all_x).max(),
            abs(all_y).max(),
        )

        if radius == 0:
            radius = 1.0

        axes.set_xlim(
            -radius - x_pad,
            radius + x_pad,
        )
        axes.set_ylim(
            -radius - y_pad,
            radius + y_pad,
        )

        axes.scatter(
            coords[0, 0],
            coords[0, 1],
            c="r",
            zorder=100,
        )

        axes.set_aspect("equal")
        axes.set_xticks([])
        axes.set_yticks([])

        axes.spines["top"].set_visible(False)
        axes.spines["right"].set_visible(False)
        axes.spines["bottom"].set_visible(False)
        axes.spines["left"].set_visible(False)

        return axes

    if edges.size == 0:
        first_x = coords[0, 0]
        first_y = coords[0, 1]

        axes.set_xlim(
            first_x - x_pad,
            first_x + x_pad,
        )
        axes.set_ylim(
            first_y - y_pad,
            first_y + y_pad,
        )
    else:
        segments = _compute_segments(
            edges,
            branch,
            depth,
            orientation,
            edge_style,
            arc_resolution,
        )

        axes.add_collection(
            LineCollection(
                segments,
                **line_kwargs,
            )
        )

        if align_leaves:
            extension_segments = _compute_leaf_extension_segments(
                branch, depth, aligned_depth, leaf_mask, orientation
            )

            if extension_segments.shape[0]:
                axes.add_collection(
                    LineCollection(
                        extension_segments,
                        **leaf_line_kwargs,
                    )
                )

        all_points = vstack(
            (
                coords_bounds[edges[:, 0]],
                coords_bounds[edges[:, 1]],
            )
        )

        axes.set_xlim(
            all_points[:, 0].min() - x_pad,
            all_points[:, 0].max() + x_pad,
        )
        axes.set_ylim(
            all_points[:, 1].min() - y_pad,
            all_points[:, 1].max() + y_pad,
        )

    if orientation == "vertical":
        if root_position == "top":
            axes.yaxis.set_inverted(True)

        axes.set_yticks(
            arange(
                0,
                depth.max() + 1,
                2,
            )
        )

        if ylabel:
            axes.set_ylabel("Tree Depth")
        else:
            axes.set_ylabel("")

    elif orientation == "horizontal":
        if root_position == "top":
            axes.xaxis.set_inverted(True)

        axes.set_xticks(
            arange(
                0,
                depth.max() + 1,
                2,
            )
        )

        axes.set_xlabel("Tree Depth")

        if ylabel:
            axes.set_ylabel("Tree Branching Position")
        else:
            axes.set_ylabel("")

    axes.scatter(
        coords[0, 0],
        coords[0, 1],
        c="r",
        zorder=100,
    )

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    # The bottom spine carries the depth axis for horizontal orientation,
    # so it must stay visible there; for vertical orientation the x-axis
    # is purely topological and the spine is hidden.
    if orientation != "horizontal":
        axes.spines["bottom"].set_visible(False)
        axes.set_xticks([])

    axes.set_aspect("auto")
    axes.set_axis_off()

    return axes
