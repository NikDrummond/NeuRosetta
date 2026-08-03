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
    count_roots,
    count_nodes,
    count_edges,
    count_leaves,
    count_branches,
    count_transitive_nodes,
    get_node_depth,
    get_node_coordinates,
    get_edge_coordinates,
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
)

from ..ops.forest_ops import (
    align_forest,
)

from ..ops.plotting import Viewer

from ..ops.units import get_units, harmonize_forest_units, ensure_forest_units, convert_units

from ..io import (
    export_swc,
    save,
)


# def _forest_op(fn: Callable) -> Callable:
#     """Create a Forest method that applies `fn` to every tree.

#     Extra per-tree kwargs are forwarded via **func_kwargs.

#     Parameters
#     ----------
#     fn : Callable
#         Function to apply to each tree.

#     Returns
#     -------
#     Callable
#         Wrapped method that applies the function to all trees.
#     """

#     def method(
#         self,
#         parallel: bool = True,
#         max_workers: int = 4,
#         progress: bool = True,
#         bind: bool = False,
#         **func_kwargs,
#     ):
#         return self.apply(
#             fn,
#             **func_kwargs,
#             parallel=parallel,
#             max_workers=max_workers,
#             show_progress=progress,
#             bind=bind,
#         )

#     method.__name__ = getattr(fn, "__name__", repr(fn))
#     method.__doc__ = f"Apply :func:`{method.__name__}` to every tree in the forest."
#     return wraps(fn)(method)

def _add_global_parameter(doc: str | None) -> str:
    """Add the global_ parameter to a NumPy-style docstring."""

    if not doc:
        return (
            "Apply the operation to the forest.\n\n"
            "Parameters\n"
            "----------\n"
            "global_ : bool, default=False\n"
            "    If False (default), apply independently to each tree.\n"
            "    If True, use the forest-wide implementation.\n"
        )

    doc = cleandoc(doc)

    parameter = (
        "global_ : bool, default=False\n"
        "    If False (default), apply independently to each tree.\n"
        "    If True, use the forest-wide implementation.\n"
    )

    marker = "Parameters\n----------"

    if marker in doc:
        return doc.replace(
            marker,
            marker + "\n" + parameter,
            1,
        )

    # No Parameters section exists; append one
    return (
        doc
        + "\n\n"
        + marker
        + "\n"
        + parameter
    )


def _forest_op(
    fn: Callable,
    *,
    global_fn: Callable | None = None,
) -> Callable:
    """Create a Forest method that applies ``fn`` to every tree.

    Parameters
    ----------
    fn : Callable
        Function applied independently to each tree.

    global_fn : Callable, optional
        Function applied to the whole forest when ``global_=True``.

    Returns
    -------
    Callable
        Wrapped Forest method.
    """

    @wraps(fn)
    def method(
        self,
        *,
        global_: bool = False,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        bind: bool = False,
        **func_kwargs,
    ):
        if global_:
            if global_fn is None:
                raise ValueError(
                    f"{fn.__name__} does not provide a global implementation."
                )

            return global_fn(
                self,
                **func_kwargs,
            )

        return self.apply(
            fn,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
            bind=bind,
        )

    if global_fn is not None:
        method.__doc__ = _add_global_parameter(fn.__doc__)

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

    # --- coordinates ---
    get_node_coordinates = _forest_op(get_node_coordinates)
    """Get node coordinates for all trees."""

    get_edge_coordinates = _forest_op(get_edge_coordinates)
    """Get edge coordinates for all trees."""

    # --- distances ---
    get_edge_length = _forest_op(get_edge_length)
    """Get edge lengths for all trees."""

    get_total_cable_length = _forest_op(get_total_cable_length)
    """Get total cable length for all trees."""

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