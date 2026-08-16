"""Forest (collection of Trees) Container class.

This module provides the Forest class which is a container for multiple Tree
objects, with batch operations that can run in parallel.
"""

from collections.abc import Callable
from functools import wraps
from inspect import cleandoc, signature

from ..config.vedo_settings import is_notebook_vedo_backend
from ..core import _Forest
from ..io import (
    export_swc,
    save,
)
from ..ops.forest_ops import (
    align_forest,
    align_forest_to_vector,
    apply_rotation_steps_to_forest,
    center_forest_at_centroid,
    coordinate_extent_along_axis_forest,
    coordinate_mean_absolute_along_axis_forest,
    coordinate_mean_along_axis_forest,
    coordinate_minmax_along_axis_forest,
    coordinate_projection_moments_forest,
    coordinate_rms_along_axis_forest,
    coordinate_std_along_axis_forest,
    coordinate_variance_along_axis_forest,
    fit_circle_forest,
    fit_line_forest,
    fit_plane_forest,
    fit_sphere_forest,
    forest_pca,
    forest_summary,
    forest_summary_table,
    get_forest_convex_hull,
    get_forest_convex_hull_volume,
    recenter_forest,
    rotate_forest,
    rotate_forest_about,
    scale_forest,
    scale_forest_about,
    scale_forest_along_pca,
    translate_forest,
)
from ..ops.plotting import Viewer
from ..ops.tree_graphs import (
    align_coordinates,
    align_coordinates_to_vector,
    apply_rotation_steps_to_coordinates,
    center_coordinates_at_centroid,
    center_coordinates_at_root,
    check_reduced,
    coordinate_extent_along_axis,
    coordinate_mean_absolute_along_axis,
    coordinate_mean_along_axis,
    coordinate_minmax_along_axis,
    coordinate_pca,
    coordinate_projection_moments,
    coordinate_rms_along_axis,
    coordinate_std_along_axis,
    coordinate_variance_along_axis,
    count_bifurcations,
    count_branches,
    count_core_nodes,
    count_edges,
    count_leaves,
    count_nodes,
    count_roots,
    count_sections,
    count_transitive_nodes,
    fit_circle,
    fit_line,
    fit_plane,
    fit_sphere,
    get_bifurcation_angle_sums,
    get_bifurcation_angles,
    get_bifurcation_deihedral_beta,
    get_bifurcation_indices,
    get_binary_ratio,
    get_branch_indices,
    get_convex_hull,
    get_convex_hull_volume,
    get_core_indices,
    get_degree_distribution,
    get_degrees,
    get_edge_angle_variance,
    get_edge_angles,
    get_edge_coordinates,
    get_edge_indices,
    get_edge_length,
    get_leaf_indices,
    get_max_depth,
    get_max_subtree_node,
    get_max_width,
    get_mean_depth,
    get_mean_edge_angle,
    get_mean_width,
    get_median_depth,
    get_median_width,
    get_node_coordinates,
    get_node_depth,
    get_partition_asymmetry,
    get_radial_angle,
    get_root_coordinate,
    get_root_index,
    get_subtree,
    get_subtree_scores,
    get_total_cable_length,
    get_tree_widths,
    has_property,
    recenter_coordinates,
    reduce_tree,
    rotate_coordinates,
    rotate_coordinates_about,
    scale_coordinates,
    scale_coordinates_about,
    scale_coordinates_along_pca,
    translate_coordinates,
    update_reduced,
)
from ..ops.units import (
    convert_units,
    ensure_forest_units,
    get_units,
    harmonize_forest_units,
    snap_voxel_coordinates,
)

# apply and global method doc strings
_FOREST_EXECUTION_PARAMETERS = """
bind : bool, optional
    Forward ``bind=True`` to the underlying tree operation when it accepts
    a ``bind`` parameter, writing results in place and returning ``None``.
    By default False.

parallel : bool or None, optional
    Run calls in a thread pool. When None, use the global forest-scoped
    setting (default True). Ignored when ``global_=True``.

max_workers : int or None, optional
    Maximum worker threads when *parallel* is True. When None, use the
    global setting (executor default when unset). Ignored when
    ``global_=True``.

show_progress : bool or None, optional
    Show a tqdm progress bar. When None, use the global setting
    (default False). Ignored when ``global_=True``.

merge_axis : int | None, optional
    Axis along which to concatenate ndarray results, or ``0`` to
    flatten list results. When None, return a plain list of per-tree
    results. Ignored when ``global_=True``. By default None.
""".strip()

_GLOBAL_PARAMETER = """
global_ : bool, default=False
    If False (default), apply the operation independently to each tree via
    :meth:`~neurosetta.api.forest_class.Forest.apply`.
    If True, use the forest-wide implementation instead.
""".strip()


def _strip_tree_parameter(doc: str) -> str:
    """Remove the implicit ``tree`` argument from a tree-op docstring."""

    lines = doc.splitlines()
    out: list[str] = []
    skip = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("tree :"):
            skip = True
            continue
        if skip:
            if stripped == "" or (line and not line[0].isspace()):
                skip = False
            else:
                continue
        out.append(line)

    return "\n".join(out).rstrip()


def _add_forest_parameters(doc: str | None) -> str:
    """Add Forest-specific parameters to a NumPy-style docstring."""

    parameters = _GLOBAL_PARAMETER + "\n\n" + _FOREST_EXECUTION_PARAMETERS
    marker = "Parameters\n----------"

    if not doc:
        return f"Apply the operation to every tree in the forest.\n\n{marker}\n{parameters}"

    doc = _strip_tree_parameter(cleandoc(doc))

    if marker in doc:
        return doc.replace(marker, f"{marker}\n{parameters}", 1)

    return f"{doc}\n\n{marker}\n{parameters}"


def _forest_op(
    fn: Callable,
    *,
    global_fn: Callable | None = None,
) -> Callable:
    """Create a Forest method that applies ``fn`` to every tree."""

    @wraps(
        fn,
        assigned=("__module__", "__name__", "__qualname__", "__doc__"),
    )
    def method(
        self,
        *,
        global_: bool = False,
        parallel: bool | None = None,
        max_workers: int | None = None,
        show_progress: bool | None = None,
        bind: bool = False,
        merge_axis: int | None = None,
        **func_kwargs,
    ):
        if global_:
            if global_fn is None:
                raise ValueError(f"{fn.__name__} does not provide a global implementation.")

            kwargs = dict(func_kwargs)
            if "bind" in signature(global_fn).parameters:
                kwargs["bind"] = bind
            return global_fn(self, **kwargs)

        return self.apply(
            fn,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
            bind=bind,
            merge_axis=merge_axis,
        )

    method.__doc__ = _add_forest_parameters(fn.__doc__)

    return method


class Forest(_Forest):
    """Container for multiple Tree objects with batch operations.

    Parameters
    ----------
    trees : list[Tree] | Forest
        List of Tree objects or another Forest.

    Notes
    -----
    Batch methods mirror the Tree API names and apply the corresponding
    tree operation to every tree in the forest (optionally in parallel).
    """

    __slots__ = ()

    def __init__(self, trees):
        super().__init__(trees=trees)

    def __contains__(self, item) -> bool:
        """Return True when *item* is a tree ID present in the forest.

        Tree objects are matched by ``ID``, not object identity.
        """
        return super().__contains__(item)

    # --- node indices ---
    get_root_index = _forest_op(get_root_index)

    get_leaf_indices = _forest_op(get_leaf_indices)

    get_branch_indices = _forest_op(get_branch_indices)

    get_core_indices = _forest_op(get_core_indices)

    get_edge_indices = _forest_op(get_edge_indices)

    get_bifurcation_indices = _forest_op(get_bifurcation_indices)

    # --- counts ---
    count_roots = _forest_op(count_roots)

    count_edges = _forest_op(count_edges)

    count_nodes = _forest_op(count_nodes)

    count_leaves = _forest_op(count_leaves)

    count_branches = _forest_op(count_branches)

    count_transitive_nodes = _forest_op(count_transitive_nodes)

    count_sections = _forest_op(count_sections)

    count_bifurcations = _forest_op(count_bifurcations)

    count_core_nodes = _forest_op(count_core_nodes)

    # --- coordinates ---
    get_node_coordinates = _forest_op(get_node_coordinates)

    get_edge_coordinates = _forest_op(get_edge_coordinates)

    get_root_coordinate = _forest_op(get_root_coordinate)

    coordinate_pca = _forest_op(coordinate_pca, global_fn=forest_pca)

    get_convex_hull = _forest_op(get_convex_hull, global_fn=get_forest_convex_hull)

    get_convex_hull_volume = _forest_op(
        get_convex_hull_volume, global_fn=get_forest_convex_hull_volume
    )

    coordinate_mean_along_axis = _forest_op(
        coordinate_mean_along_axis,
        global_fn=coordinate_mean_along_axis_forest,
    )

    coordinate_variance_along_axis = _forest_op(
        coordinate_variance_along_axis,
        global_fn=coordinate_variance_along_axis_forest,
    )

    coordinate_std_along_axis = _forest_op(
        coordinate_std_along_axis,
        global_fn=coordinate_std_along_axis_forest,
    )

    coordinate_minmax_along_axis = _forest_op(
        coordinate_minmax_along_axis,
        global_fn=coordinate_minmax_along_axis_forest,
    )

    coordinate_extent_along_axis = _forest_op(
        coordinate_extent_along_axis,
        global_fn=coordinate_extent_along_axis_forest,
    )

    coordinate_rms_along_axis = _forest_op(
        coordinate_rms_along_axis,
        global_fn=coordinate_rms_along_axis_forest,
    )

    coordinate_mean_absolute_along_axis = _forest_op(
        coordinate_mean_absolute_along_axis,
        global_fn=coordinate_mean_absolute_along_axis_forest,
    )

    coordinate_projection_moments = _forest_op(
        coordinate_projection_moments,
        global_fn=coordinate_projection_moments_forest,
    )

    # --- distances ---
    get_edge_length = _forest_op(get_edge_length)

    get_total_cable_length = _forest_op(get_total_cable_length)

    # --- Tree geometry ---
    get_edge_angles = _forest_op(get_edge_angles)

    get_mean_edge_angle = _forest_op(get_mean_edge_angle)

    get_edge_angle_variance = _forest_op(get_edge_angle_variance)

    get_radial_angle = _forest_op(get_radial_angle)

    get_bifurcation_angles = _forest_op(get_bifurcation_angles)

    get_bifurcation_angle_sums = _forest_op(get_bifurcation_angle_sums)

    get_bifurcation_deihedral_beta = _forest_op(get_bifurcation_deihedral_beta)

    # --- Shape Fitting ---
    fit_sphere = _forest_op(fit_sphere, global_fn=fit_sphere_forest)

    fit_line = _forest_op(fit_line, global_fn=fit_line_forest)

    fit_plane = _forest_op(fit_plane, global_fn=fit_plane_forest)

    fit_circle = _forest_op(fit_circle, global_fn=fit_circle_forest)

    # --- topology ---
    get_node_depth = _forest_op(get_node_depth)

    get_max_depth = _forest_op(get_max_depth)

    get_mean_depth = _forest_op(get_mean_depth)

    get_median_depth = _forest_op(get_median_depth)

    get_tree_widths = _forest_op(get_tree_widths)

    get_max_width = _forest_op(get_max_width)

    get_mean_width = _forest_op(get_mean_width)

    get_median_width = _forest_op(get_median_width)

    get_binary_ratio = _forest_op(get_binary_ratio)

    get_degrees = _forest_op(get_degrees)

    get_degree_distribution = _forest_op(get_degree_distribution)

    # --- checks / updates ---
    has_property = _forest_op(has_property)

    is_reduced = _forest_op(check_reduced)

    update_reduced = _forest_op(update_reduced)

    # --- editing ---
    reduce_tree = _forest_op(reduce_tree)

    # --- subtrees ---
    get_subtree_scores = _forest_op(get_subtree_scores)

    get_max_subtree_node = _forest_op(get_max_subtree_node)

    get_subtree = _forest_op(get_subtree)

    get_partition_asymmetry = _forest_op(get_partition_asymmetry)

    # --- transformations ---
    align_coordinates = _forest_op(
        align_coordinates,
        global_fn=align_forest,
    )

    translate_coordinates = _forest_op(
        translate_coordinates,
        global_fn=translate_forest,
    )

    center_coordinates_at_centroid = _forest_op(
        center_coordinates_at_centroid,
        global_fn=center_forest_at_centroid,
    )

    center_coordinates_at_root = _forest_op(
        center_coordinates_at_root,
    )

    recenter_coordinates = _forest_op(
        recenter_coordinates,
        global_fn=recenter_forest,
    )

    rotate_coordinates = _forest_op(
        rotate_coordinates,
        global_fn=rotate_forest,
    )

    rotate_coordinates_about = _forest_op(
        rotate_coordinates_about,
        global_fn=rotate_forest_about,
    )

    scale_coordinates = _forest_op(
        scale_coordinates,
        global_fn=scale_forest,
    )

    scale_coordinates_about = _forest_op(
        scale_coordinates_about,
        global_fn=scale_forest_about,
    )

    align_coordinates_to_vector = _forest_op(
        align_coordinates_to_vector,
        global_fn=align_forest_to_vector,
    )

    scale_coordinates_along_pca = _forest_op(
        scale_coordinates_along_pca,
        global_fn=scale_forest_along_pca,
    )

    apply_rotation_steps_to_coordinates = _forest_op(
        apply_rotation_steps_to_coordinates,
        global_fn=apply_rotation_steps_to_forest,
    )

    align_forest = align_coordinates
    """Alias for :meth:`align_coordinates`."""

    # --- units ---
    get_units = _forest_op(get_units)

    convert_units = _forest_op(convert_units)

    snap_voxel_coordinates = _forest_op(snap_voxel_coordinates)
    """Snap continuous voxel coordinates to integer grid indices on every tree."""

    harmonize_forest_units = harmonize_forest_units
    """Convert all trees to a common unit system, warning on mixed units."""

    ensure_forest_units = ensure_forest_units
    """Harmonize units and require spatial units on every tree."""

    # --- saving ---
    export_forest_to_swc = export_swc
    """Export forest to SWC format."""

    save_forest = save
    """Save forest to file."""

    summary = forest_summary
    """Return a formatted summary table for all trees in the forest (HTML in notebooks)."""

    summary_table = forest_summary_table
    """Return per-tree summary metrics as a DataFrame."""

    # --- plotting ---
    def show_3d(self, return_viewer: bool = False, **kwargs):
        """Show 3D interactive plot of all trees in the forest.

        Parameters
        ----------
        return_viewer : bool, optional
            If True, return the Viewer object instead of showing it.
            By default False.
        **kwargs : dict
            Additional keyword arguments passed to Viewer.add_forest().

        Returns
        -------
        Viewer | Any | None
            The Viewer if *return_viewer* is True. For notebook vedo backends
            (e.g. ``k3d``), the inline widget from ``show()``. Otherwise None
            for desktop ``vtk`` after the window is closed.
        """
        v = Viewer()
        v.add_forest(self, **kwargs)
        if return_viewer:
            return v
        shown = v.show()
        if is_notebook_vedo_backend():
            return shown
        shown.close()

    # --- core container helpers inherited from _Forest ---
    # apply / filter / map / build_3d
    # append / extend / insert / remove / remove_id / pop / clear
    # by_id / ids / copy / __contains__ / __getitem__ / __len__ / __iter__ / __add__
    # get_meta / set_meta / del_meta / has_meta / list_meta / meta_summary / meta_indices
    # list_properties — see _Forest and docs/api/forest.md
