"""Forest (collection of Trees) Container class.

This module provides the Forest class which is a container for multiple Tree
objects, with batch operations that can run in parallel.
"""

from functools import wraps
from typing import Callable

from ..core import _Forest

from ..ops.tree_graphs import (
    get_root,
    get_leaves,
    get_branches,
    get_core_indices,
    get_edges,
    count_tree_roots,
    count_tree_nodes,
    count_tree_edges,
    count_tree_leaves,
    count_tree_branches,
    count_tree_transitive_nodes,
    compute_tree_depths,
    tree_node_coordinates,
    tree_edge_coordinates,
    check_reduced,
    update_reduced,
    tree_has_property,
    euclidean_edge_length,
    total_cable_length,
    get_node_degrees,
    tree_degree_distribution,
    reduce_tree,
    score_subtrees,
    max_subtree_ind,
    extract_subtree,
    node_partition_asymmetry,
)

from ..ops.plotting import Viewer

from ..io import (
    export_swc,
    save,
)


def _forest_op(fn: Callable) -> Callable:
    """Create a Forest method that applies `fn` to every tree.

    Extra per-tree kwargs are forwarded via **func_kwargs.

    Parameters
    ----------
    fn : Callable
        Function to apply to each tree.

    Returns
    -------
    Callable
        Wrapped method that applies the function to all trees.
    """

    def method(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        bind: bool = False,
        **func_kwargs,
    ):
        return self.apply(
            fn,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
            bind=bind,
        )

    method.__name__ = getattr(fn, "__name__", repr(fn))
    method.__doc__ = f"Apply :func:`{method.__name__}` to every tree in the forest."
    return wraps(fn)(method)


class Forest(_Forest):
    """Container for multiple Tree objects with batch operations.

    Parameters
    ----------
    trees : list[Tree] | Forest
        List of Tree objects or another Forest.

    Notes
    -----
    All methods prefixed with `get_forest_` or `count_forest_` apply the
    corresponding tree operation to every tree in the forest and can run
    in parallel.
    """

    def __init__(self, trees):
        super().__init__(trees=trees)

    # --- node indices ---
    get_forest_roots = _forest_op(get_root)
    """Get root indices for all trees."""

    get_forest_leaves = _forest_op(get_leaves)
    """Get leaf indices for all trees."""

    get_forest_branches = _forest_op(get_branches)
    """Get branch indices for all trees."""

    get_forest_core_indices = _forest_op(get_core_indices)
    """Get core indices for all trees."""

    get_forest_edge_indices = _forest_op(get_edges)
    """Get edge indices for all trees."""

    # --- counts ---
    count_forest_roots = _forest_op(count_tree_roots)
    """Count roots in all trees."""

    count_forest_edges = _forest_op(count_tree_edges)
    """Count edges in all trees."""

    count_forest_nodes = _forest_op(count_tree_nodes)
    """Count nodes in all trees."""

    count_forest_leaves = _forest_op(count_tree_leaves)
    """Count leaves in all trees."""

    count_forest_branches = _forest_op(count_tree_branches)
    """Count branches in all trees."""

    count_forest_transitive_nodes = _forest_op(count_tree_transitive_nodes)
    """Count transitive nodes in all trees."""

    # --- coordinates ---
    forest_node_coordinates = _forest_op(tree_node_coordinates)
    """Get node coordinates for all trees."""

    forest_edge_coordinates = _forest_op(tree_edge_coordinates)
    """Get edge coordinates for all trees."""

    # --- distances ---
    forest_edge_lengths = _forest_op(euclidean_edge_length)
    """Get edge lengths for all trees."""

    forest_total_cable_length = _forest_op(total_cable_length)
    """Get total cable length for all trees."""

    # --- topology ---
    forest_node_depths = _forest_op(compute_tree_depths)
    """Get node depths for all trees."""

    forest_degree_arrays = _forest_op(get_node_degrees)
    """Get degree arrays for all trees."""

    forest_degree_distributions = _forest_op(tree_degree_distribution)
    """Get degree distributions for all trees."""

    # --- checks / updates ---
    forest_has_property = _forest_op(tree_has_property)
    """Check if all trees have a property."""

    is_reduced = _forest_op(check_reduced)
    """Check if all trees are reduced."""

    forest_update_reduced = _forest_op(update_reduced)
    """Update reduced flag for all trees."""

    # --- editing ---
    reduce_forest = _forest_op(reduce_tree)
    """Reduce all trees in the forest."""

    # --- subtrees ---
    get_forest_subtree_scores = _forest_op(score_subtrees)
    """Get subtree scores for all trees."""

    get_forest_max_subtree_index = _forest_op(max_subtree_ind)
    """Get max subtree index for all trees."""

    convert_forest_to_subtrees = _forest_op(extract_subtree)
    """Extract subtrees for all trees."""

    get_forest_node_partition_asymmetry = _forest_op(node_partition_asymmetry)
    """Get partition asymmetry for all trees."""

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