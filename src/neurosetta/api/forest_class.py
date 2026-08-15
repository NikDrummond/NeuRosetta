"""Forest (collection of Trees) Container class.

This module provides the Forest class which is a container for multiple Tree
objects, with batch operations that can run in parallel.
"""

from collections.abc import Callable
from functools import wraps
from inspect import cleandoc, signature

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
    get_max_subtree_node,
    get_mean_edge_angle,
    get_node_coordinates,
    get_node_depth,
    get_partition_asymmetry,
    get_radial_angle,
    get_root_coordinate,
    get_root_index,
    get_subtree,
    get_subtree_scores,
    get_total_cable_length,
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
from ..ops.units import convert_units, ensure_forest_units, get_units, harmonize_forest_units

# apply and global method doc strings
_FOREST_EXECUTION_PARAMETERS = """
parallel : bool
    Run calls in a thread pool. By default True.

max_workers : int | None
    Maximum worker threads when *parallel* is True. By default None
    (executor default).

show_progress : bool, optional
    Show a tqdm progress bar. By default False.

merge_axis : int | None, optional
    Axis along which to concatenate ndarray results, or ``0`` to
    flatten list results. When None, return a plain list of per-tree
    results. By default None.
""".strip()

_GLOBAL_PARAMETER = """
global_ : bool, default=False
    If False (default), apply the operation independently to each tree.
    If True, use the forest-wide implementation.
""".strip()


def _add_forest_parameters(doc: str | None) -> str:
    """Add Forest-specific parameters to a NumPy-style docstring."""

    parameters = _GLOBAL_PARAMETER + "\n\n" + _FOREST_EXECUTION_PARAMETERS
    marker = "Parameters\n----------"

    if not doc:
        return f"Apply the operation to the forest.\n\n{marker}\n{parameters}"

    doc = cleandoc(doc)

    if marker in doc:
        return doc.replace(marker, f"{marker}\n{parameters}", 1)

    return f"{doc}\n\n{marker}\n{parameters}"


def _forest_op(
    fn: Callable,
    *,
    global_fn: Callable | None = None,
) -> Callable:
    """Create a Forest method that applies ``fn`` to every tree."""

    @wraps(fn)
    def method(
        self,
        *,
        global_: bool = False,
        parallel: bool = True,
        max_workers: int = 4,
        show_progress: bool = True,
        bind: bool = False,
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

    # --- node indices ---
    get_root_index = _forest_op(get_root_index)
    """Get root indices for all trees."""

    get_leaf_indices = _forest_op(get_leaf_indices)
    """Get leaf indices for all trees."""

    get_branch_indices = _forest_op(get_branch_indices)
    """Get branch indices for all trees."""

    get_core_indices = _forest_op(get_core_indices)
    """Get core indices for all trees."""

    get_edge_indices = _forest_op(get_edge_indices)
    """Get edge indices for all trees."""

    get_bifurcation_indices = _forest_op(get_bifurcation_indices)
    """get indices of bifurcation nodes for all trees"""

    # --- counts ---
    count_roots = _forest_op(count_roots)
    """Count roots in all trees."""

    count_edges = _forest_op(count_edges)
    """Count edges in all trees."""

    count_nodes = _forest_op(count_nodes)
    """Count nodes in all trees."""

    count_leaves = _forest_op(count_leaves)
    """Count leaves in all trees."""

    count_branches = _forest_op(count_branches)
    """Count branches in all trees."""

    count_transitive_nodes = _forest_op(count_transitive_nodes)
    """Count transitive nodes in all trees."""

    count_sections = _forest_op(count_sections)
    """Count the number of sectiosn in all trees"""

    count_bifurcations = _forest_op(count_bifurcations)
    """Count the number of bifurcating nodes in all trees"""

    # --- coordinates ---
    get_node_coordinates = _forest_op(get_node_coordinates)
    """Get node coordinates for all trees."""

    get_edge_coordinates = _forest_op(get_edge_coordinates)
    """Get edge coordinates for all trees."""

    get_root_coordinate = _forest_op(get_root_coordinate)
    """Get the root coorrdinate for all trees"""

    coordinate_pca = _forest_op(coordinate_pca, global_fn=forest_pca)

    get_convex_hull = _forest_op(get_convex_hull, global_fn=get_forest_convex_hull)

    get_convex_hull_volume = _forest_op(
        get_convex_hull_volume, global_fn=get_forest_convex_hull_volume
    )

    coordinate_mean_along_axis = _forest_op(
        coordinate_mean_along_axis,
        global_fn=coordinate_mean_along_axis_forest,
    )
    """Compute mean projection of node coordinates onto an axis."""

    coordinate_variance_along_axis = _forest_op(
        coordinate_variance_along_axis,
        global_fn=coordinate_variance_along_axis_forest,
    )
    """Compute variance of node coordinate projections onto an axis."""

    coordinate_std_along_axis = _forest_op(
        coordinate_std_along_axis,
        global_fn=coordinate_std_along_axis_forest,
    )
    """Compute standard deviation of node coordinate projections onto an axis."""

    coordinate_minmax_along_axis = _forest_op(
        coordinate_minmax_along_axis,
        global_fn=coordinate_minmax_along_axis_forest,
    )
    """Compute minimum and maximum projections onto an axis."""

    coordinate_extent_along_axis = _forest_op(
        coordinate_extent_along_axis,
        global_fn=coordinate_extent_along_axis_forest,
    )
    """Compute extent of node coordinate projections onto an axis."""

    coordinate_rms_along_axis = _forest_op(
        coordinate_rms_along_axis,
        global_fn=coordinate_rms_along_axis_forest,
    )
    """Compute root-mean-square projection onto an axis."""

    coordinate_mean_absolute_along_axis = _forest_op(
        coordinate_mean_absolute_along_axis,
        global_fn=coordinate_mean_absolute_along_axis_forest,
    )
    """Compute mean absolute projection onto an axis."""

    coordinate_projection_moments = _forest_op(
        coordinate_projection_moments,
        global_fn=coordinate_projection_moments_forest,
    )
    """Compute summary projection moments onto an axis."""

    # --- distances ---
    get_edge_length = _forest_op(get_edge_length)
    """Get edge lengths for all trees."""

    get_total_cable_length = _forest_op(get_total_cable_length)
    """Get total cable length for all trees."""

    # --- Tree geometry ---
    get_edge_angles = _forest_op(get_edge_angles)
    """Get the angle of edges from between vector from perspective vector"""

    get_mean_edge_angle = _forest_op(get_mean_edge_angle)
    """Get the mean angle of edges in neuron and between vector from perspective vector"""

    get_edge_angle_variance = _forest_op(get_edge_angle_variance)
    """Get the angle variance of edges in neuron and between vector from perspective vector"""

    get_radial_angle = _forest_op(get_radial_angle)
    """Get radial angle of edges from root node"""

    get_bifurcation_angles = _forest_op(get_bifurcation_angles)
    """Get the angles between edges at bifurcations nodes"""

    get_bifurcation_angle_sums = _forest_op(get_bifurcation_angle_sums)
    """Get the sum of angles between edges at bifurcations points"""

    get_bifurcation_deihedral_beta = _forest_op(get_bifurcation_deihedral_beta)
    """Get Dihedral beta angle for all bifurcations (measure of planarity)"""

    # --- Shape Fitting ---
    fit_sphere = _forest_op(fit_sphere, global_fn=fit_sphere_forest)
    """Fit a sphere to individual neuron coordinates or all coordinates"""

    fit_line = _forest_op(fit_line, global_fn=fit_line_forest)
    """Fit a line to individual neuron coordinates or all coordinates"""

    fit_plane = _forest_op(fit_plane, global_fn=fit_plane_forest)
    """Fit a plane to individual neuron coordinates or all coordinates"""

    fit_circle = _forest_op(fit_circle, global_fn=fit_circle_forest)
    """Fit a circle to individual neuron coordinates or all coordinates"""

    # --- topology ---
    get_node_depth = _forest_op(get_node_depth)
    """Get node depths for all trees."""

    get_degrees = _forest_op(get_degrees)
    """Get degree arrays for all trees."""

    get_degree_distribution = _forest_op(get_degree_distribution)
    """Get degree distributions for all trees."""

    # --- checks / updates ---
    has_property = _forest_op(has_property)
    """Check if all trees have a property."""

    is_reduced = _forest_op(check_reduced)
    """Check if all trees are reduced."""

    update_reduced = _forest_op(update_reduced)
    """Update reduced flag for all trees."""

    # --- editing ---
    reduce_tree = _forest_op(reduce_tree)
    """Reduce all trees in the forest."""

    # --- subtrees ---
    get_subtree_scores = _forest_op(get_subtree_scores)
    """Get subtree scores for all trees."""

    get_max_subtree_node = _forest_op(get_max_subtree_node)
    """Get max subtree index for all trees."""

    get_subtree = _forest_op(get_subtree)
    """Extract subtrees for all trees."""

    get_partition_asymmetry = _forest_op(get_partition_asymmetry)
    """Get partition asymmetry for all trees."""

    # --- transformations ---
    align_coordinates = _forest_op(
        align_coordinates,
        global_fn=align_forest,
    )
    """PCA-align node coordinates to canonical axes."""

    translate_coordinates = _forest_op(
        translate_coordinates,
        global_fn=translate_forest,
    )
    """Translate all node coordinates by a fixed displacement."""

    center_coordinates_at_centroid = _forest_op(
        center_coordinates_at_centroid,
        global_fn=center_forest_at_centroid,
    )
    """Translate trees so their node centroids lie at the origin."""

    center_coordinates_at_root = _forest_op(
        center_coordinates_at_root,
    )
    """Translate each tree so its root node lies at the origin."""

    recenter_coordinates = _forest_op(
        recenter_coordinates,
        global_fn=recenter_forest,
    )
    """Translate trees so a specified point becomes the origin."""

    rotate_coordinates = _forest_op(
        rotate_coordinates,
        global_fn=rotate_forest,
    )
    """Rotate all node coordinates about an axis through the origin."""

    rotate_coordinates_about = _forest_op(
        rotate_coordinates_about,
        global_fn=rotate_forest_about,
    )
    """Rotate all node coordinates about an axis through a point."""

    scale_coordinates = _forest_op(
        scale_coordinates,
        global_fn=scale_forest,
    )
    """Uniformly scale node coordinates about a point."""

    scale_coordinates_about = _forest_op(
        scale_coordinates_about,
        global_fn=scale_forest_about,
    )
    """Scale node coordinates independently along each axis about a point."""

    align_coordinates_to_vector = _forest_op(
        align_coordinates_to_vector,
        global_fn=align_forest_to_vector,
    )
    """Rotate trees so a source direction aligns with a target direction."""

    scale_coordinates_along_pca = _forest_op(
        scale_coordinates_along_pca,
        global_fn=scale_forest_along_pca,
    )
    """Scale node coordinates along PCA axes."""

    apply_rotation_steps_to_coordinates = _forest_op(
        apply_rotation_steps_to_coordinates,
        global_fn=apply_rotation_steps_to_forest,
    )
    """Apply two precomputed axis-angle rotations to all node coordinates."""

    align_forest = align_coordinates
    """Alias for :meth:`align_coordinates`."""

    # --- units ---
    get_units = _forest_op(get_units)
    """Get spatial units for all trees."""

    convert_units = _forest_op(convert_units)
    """Convert spatial units for all trees."""

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
    def show_3d(self, return_viewer: bool = False, **kwargs) -> "Viewer":
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
        Viewer | None
            The Viewer object if return_viewer=True, otherwise None.
        """
        v = Viewer()
        v.add_forest(self, **kwargs)
        if return_viewer:
            return v
        v.show().close()
