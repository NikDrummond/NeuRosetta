"""Forest (collection of Trees) Container class"""

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

from ..ops.plotting.utils import _build_3d
from ..ops.plotting import Viewer

from ..io import (
    export_swc,
    save,
)

# In forest.py, before the class definition


def _forest_op(fn: Callable) -> Callable:
    """
    Creates a Forest method that applies `fn` to every tree.
    Extra per-tree kwargs are forwarded via **func_kwargs.
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
    return method


class Forest(_Forest):

    def __init__(self, trees):
        super().__init__(trees=trees)

    # --- node indices ---
    get_forest_roots = _forest_op(get_root)
    get_forest_leaves = _forest_op(get_leaves)
    get_forest_branches = _forest_op(get_branches)
    get_forest_core_indices = _forest_op(get_core_indices)
    get_forest_edge_indices = _forest_op(get_edges)

    # --- counts ---
    count_forest_roots = _forest_op(count_tree_roots)
    count_forest_edges = _forest_op(count_tree_edges)
    count_forest_nodes = _forest_op(count_tree_nodes)
    count_forest_leaves = _forest_op(count_tree_leaves)
    count_forest_branches = _forest_op(count_tree_branches)
    count_forest_transitive_nodes = _forest_op(count_tree_transitive_nodes)

    # --- coordinates ---
    forest_node_coordinates = _forest_op(tree_node_coordinates)
    forest_edge_coordinates = _forest_op(tree_edge_coordinates)

    # --- distances ---
    forest_edge_lengths = _forest_op(euclidean_edge_length)
    forest_total_cable_length = _forest_op(total_cable_length)

    # --- topology ---
    forest_node_depths = _forest_op(compute_tree_depths)  # pass root=, bind= via kwargs
    forest_degree_arrays = _forest_op(get_node_degrees)
    forest_degree_distributions = _forest_op(tree_degree_distribution)

    # --- checks / updates ---
    forest_has_property = _forest_op(tree_has_property)
    is_reduced = _forest_op(check_reduced)
    forest_update_reduced = _forest_op(update_reduced)

    # --- editing ---
    reduce_forest = _forest_op(reduce_tree)

    # --- subtrees ---
    get_forest_subtree_scores = _forest_op(score_subtrees)
    get_forest_max_subtree_index = _forest_op(max_subtree_ind)
    convert_forest_to_subtrees = _forest_op(extract_subtree)
    get_forest_node_partition_asymmetry = _forest_op(node_partition_asymmetry)

    # --- saving ---
    export_forest_to_swc = export_swc
    save_forest = save

    # --- plotting ---
    def _gen_3d(self, parallel=True, max_workers=4, progress=True) -> None:
        return self.apply(
            _build_3d,
            cache=True,
            random_c=True,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def show_3d(self, return_viewer: bool = False, **kwargs) -> "Viewer":
        v = Viewer()
        v.add_forest(self, **kwargs)
        if return_viewer:
            return v
        v.show().close()
