from graph_tool.all import Graph
from numpy import ndarray

from ..core import _Tree
from ..tree_graphs.coordinates import vertex_coordinates
from ..tree_graphs.vertex_inds import get_root, get_leaves, get_branches, get_core_inds
from ..io_utils.swc_utils import write_swc as _write_swc_func


class Tree_graph(_Tree):

    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID = ID, units = units, meta = meta, graph = graph)

    ### get node inds
    def root_index(self) -> int:
        return get_root(self)
    
    def eaf_indicies()


    ### get coordinates
    def get_coordinates(self, subset = None) -> ndarray:
        return vertex_coordinates(self, subset)

    ### saving
    def write_swc(self, fpath: str, header = None):
        
        from ..io_utils.swc_utils import write_swc
        write_swc(self, fpath, header)

### adopt doc strings
Tree_graph.write_swc.__doc__ = _write_swc_func.__doc__
Tree_graph.get_coordinates.__doc__ = vertex_coordinates.__doc__