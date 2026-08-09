"""Forest (collection of Trees) Container class.

This module provides the Forest class which is a container for multiple Tree
objects, with batch operations that can run in parallel.
"""

from functools import wraps
from inspect import cleandoc
from typing import Callable

from ..core import _Forest

from ..ops.tree_graphs import (
    get_root_index,
    get_leaf_indices,
    get_branch_indices,
    get_core_indices,
    get_edge_indices,
    get_bifurcation_indices,
    count_roots,
    count_nodes,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_nodes,
    count_sections,
    count_bifurcations,
    get_node_depth,
    get_node_coordinates,
    get_edge_coordinates,
    coordinate_pca,
    get_convex_hull,
    get_convex_hull_volume,
    check_reduced,
    update_reduced,
    has_property,
    get_edge_length,
    get_total_cable_length,
    get_degrees,
    get_degree_distribution,
    reduce_tree,
    get_subtree_scores,
    get_max_subtree_node,
    get_subtree,
    get_partition_asymmetry,
    align_tree,
    get_edge_angles,
    get_mean_edge_angle,
    get_edge_angle_variance,
    get_radial_angle,
    get_bifurcation_angles,
    get_bifurcation_angle_sums,
    get_bifurcation_deihedral_beta,
    fit_circle,
    fit_line,
    fit_plane, 
    fit_sphere,
)

from ..ops.forest_ops import (
    align_forest,
    forest_pca,
    get_forest_convex_hull,
    get_forest_convex_hull_volume,
    fit_sphere_forest,
    fit_plane_forest,
    fit_line_forest,
    fit_circle_forest,
)

from ..ops.plotting import Viewer

from ..ops.units import get_units, harmonize_forest_units, ensure_forest_units, convert_units

from ..io import (
    export_swc,
    save,
)

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
        return (
            "Apply the operation to the forest.\n\n"
            f"{marker}\n"
            f"{parameters}"
        )

    doc = cleandoc(doc)

    if marker in doc:
        return doc.replace(marker, f"{marker}\n{parameters}", 1)

    return (
        f"{doc}\n\n"
        f"{marker}\n"
        f"{parameters}"
    )


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
                raise ValueError(
                    f"{fn.__name__} does not provide a global implementation."
                )

            return global_fn(self, **func_kwargs)

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

    coordinate_pca = _forest_op(
        coordinate_pca,
        global_fn=forest_pca
    )

    get_convex_hull = _forest_op(
        get_convex_hull,
        global_fn=get_forest_convex_hull
    )

    get_convex_hull_volume = _forest_op(
        get_convex_hull_volume,
        global_fn=get_forest_convex_hull_volume
    )

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
    fit_sphere = _forest_op(
        fit_sphere,
        global_fn=fit_sphere_forest
                            )
    """Fit a sphere to individual neuron coordinates or all coordinates"""

    fit_line = _forest_op(
        fit_line,
        global_fn=fit_line_forest
                            )
    """Fit a line to individual neuron coordinates or all coordinates"""

    fit_plane = _forest_op(
        fit_plane,
        global_fn=fit_plane_forest
                            )
    """Fit a plane to individual neuron coordinates or all coordinates"""

    fit_circle = _forest_op(
        fit_circle,
        global_fn=fit_circle_forest
                            )
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
    align_forest = _forest_op(
        align_tree,
        global_fn=align_forest
    )

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