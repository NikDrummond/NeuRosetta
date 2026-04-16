"""Forest (collection of Trees) Container class"""

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
)

from ..ops.plotting.utils import _build_3d
from ..ops.plotting import Viewer

from ..io import (
    export_swc,
    save,
)


class Forest(_Forest):

    def __init__(self, trees):
        super().__init__(trees=trees)

    ### node inds
    def get_forest_roots(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_root,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def get_forest_leaves(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_leaves,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def get_forest_branches(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_branches,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def get_forest_core_indices(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_core_indices,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def get_forest_edge_indices(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_edges,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### counting
    def count_forest_roots(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_roots,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def count_forest_edges(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_edges,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def count_forest_branches(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_branches,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def count_forest_leaves(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_leaves,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def count_forest_nodes(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_nodes,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def count_forest_transitive_nodes(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            count_tree_transitive_nodes,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### coordinates
    def forest_node_coordinates(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            tree_node_coordinates,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def forest_edge_coordinates(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            tree_edge_coordinates,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### distances
    def forest_edge_lengths(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            euclidean_edge_length,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def forest_total_cable_length(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            total_cable_length,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### Topological bits

    # parallelisation not working here? (no speed up)
    def forest_node_depths(
        self,
        root: int | None = None,
        bind: bool = True,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            compute_tree_depths,
            root=root,
            bind=bind,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### degrees
    def forest_degree_arrays(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            get_node_degrees,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def forest_degree_distributions(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            tree_degree_distribution,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### saving
    export_forest_to_swc = export_swc
    save_forest = save

    ### checks
    def forest_has_property(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            tree_has_property,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def is_reduced(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            check_reduced,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### updateing metadata
    def forest_update_reduced(
        self, parallel: bool = True, max_workers: int = 4, progress: bool = True
    ) -> None:
        return self.apply(
            update_reduced,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### Editing trees
    def reduce_forest(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            reduce_tree,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### subtrees
    def get_forest_subtree_scores(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            score_subtrees,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def get_forest_max_subtree_index(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        return self.apply(
            max_subtree_ind,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def convert_forest_to_subtrees(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        **func_kwargs,
    ):
        self.apply(
            extract_subtree,
            **func_kwargs,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    ### plotting
    def _gen_3d(
        self,
        parallel: bool = True,
        max_workers: int = 4,
        progress: bool = True,
        # force_refresh: bool = False,
    ) -> None:
        return self.apply(
            _build_3d,
            # force_refresh = force_refresh,
            cache=True,
            random_c=True,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=progress,
        )

    def show_3d(self, return_viewer: bool = False, **kwargs) -> Viewer:

        v = Viewer()
        v.add_forest(self, **kwargs)

        if return_viewer:
            return v
        v.show().close()
