from ..core import _Forest

from ..tree_graphs.vertex_inds import get_root_forest, get_leaves_forest, get_branches_forest, get_core_inds_forest, get_edges_forest

from ..io_utils.swc_utils import export_swc as _write_swc_func
from ..io_utils.nr_utils import save as _save

class Forest(_Forest):

    def __init__(self, trees):
        super().__init__(trees = trees)

    ### node inds
    root_indices = get_root_forest
    leaf_indeices = get_leaves_forest
    branch_indices = get_branches_forest
    core_indices = get_core_inds_forest
    edge_indices = get_edges_forest

    ### io utils
    export_swc = _write_swc_func
    save = _save