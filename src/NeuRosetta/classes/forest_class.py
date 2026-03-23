from ..core import _Forest

from ..tree_graphs.vertex_inds import (
    get_root,
    get_leaves,
    get_branches,
    get_core_inds,
    get_edges,
)
from ..tree_graphs.counting import (
    count_roots,
    count_edges,
    count_branches,
    count_leaves,
    count_vertices,
)
from ..tree_graphs.coordinates import vertex_coordinates, edge_coordinates
from ..tree_graphs.tree_checks import is_Reduced, update_reduced, has_property

from ..tree_graphs.degrees import get_degrees, degree_distribution

from ..tree_graphs.traversals import BF_search, compute_depths, DF_search, compute_post_order

from ..io_utils.swc_utils import export_swc as _write_swc_func
from ..io_utils.nr_utils import save as _save

from ..plotting.utils import _build_3d, _random_c
from ..plotting.viewer import Viewer

class Forest(_Forest):

    def __init__(self, trees):
        super().__init__(trees = trees)

    ### node inds
    def root_indices(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_root, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def leaf_indices(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_leaves, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def branch_indices(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_branches, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def core_indices(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_core_inds, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def edge_indices(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_edges, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### counting
    def num_roots(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(count_roots, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def num_edges(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(count_edges, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def num_branches(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(count_branches, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    def num_leaves(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(count_leaves, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def num_nodes(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(count_vertices, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    ### coordinates
    def get_node_coordinates(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(vertex_coordinates, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def get_edge_coordinates(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(edge_coordinates, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### Traversals

    # def Breadth_first_search(self, visitor: BFSVisitor, init_properties: dict = {}, root: int | None = None, bind: bool = True, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
    #     return self.apply_fn(BF_search, visitor = visitor, init_properties = init_properties, root = root, bind = bind, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    # def Depth_first_search(self, visitor: DFSVisitor, init_properties: dict = {}, root: int | None = None, bind: bool = True, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
    #     return self.apply_fn(DF_search, visitor = visitor, init_properties = init_properties, root = root, bind = bind, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def Get_post_order_traversal(self, root: int | None = None, bind: bool = True, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(compute_post_order, root = root, bind = bind, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### Topological bits

    def Get_node_depths(self, root: int | None = None, bind: bool = True, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(compute_depths, root = root, bind = bind, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### degrees
    def Get_degree_arrays(self, parallel: bool = True, max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(get_degrees, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def Get_degree_distributions(self, parallel: bool = True, max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(degree_distribution, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### saving
    export_swc = _write_swc_func
    save = _save

    ### checks
    def has_property(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(has_property, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)
    
    def is_reduced(self, parallel: bool = True,  max_workers: int = 4, progress: bool = True, **func_kwargs):
        return self.apply_fn(is_Reduced, **func_kwargs, parallel = parallel, max_workers = max_workers, show_progress = progress)

    ### updateing metadata
    def update_reduced(self, parallel: bool = True, max_workers: int = 4, progress: bool = True) -> None:
        return self.foreach(update_reduced, parallel = parallel, max_workers = max_workers, show_progress = progress)


    ### plotting
    def _gen_3d(self, parallel: bool = True, max_workers: int = 4, progress: bool = True) -> None:
        return self.foreach(_build_3d, cache = True, parallel = parallel, max_workers = max_workers, show_progress = progress)

    def show_3d(self, return_viewer: bool = False, **kwargs) -> Viewer:

        v = Viewer()
        v.add_forest(self, **kwargs)

        if return_viewer:
            return v
        v.show().close()
