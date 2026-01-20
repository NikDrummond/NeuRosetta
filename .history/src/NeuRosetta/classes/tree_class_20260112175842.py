from graph_tool.all import Graph

from ..core import _Tree
from ..tree_graphs.coordinates import vertex_coordinates, edge_coordinates
from ..tree_graphs.vertex_inds import get_root, get_leaves, get_branches, get_core_inds, get_edges
from ..tree_graphs.counting import count_roots, count_edges, count_branches, count_leaves, cuont
from ..io_utils.swc_utils import write_swc as _write_swc_func


class Tree_graph(_Tree):

    def __init__(self, ID:int, units:str, meta:dict, graph:Graph) -> None:
        super().__init__(ID = ID, units = units, meta = meta, graph = graph)

    ### get node inds
    root_index = get_root
    leaf_indices = get_leaves
    branch_indices = get_branches
    core_indices = get_core_inds
    edge_indices = get_edges

    ### counting
    count_roots = count_roots


    ### coordinates
    get_node_coordinates = vertex_coordinates
    get_edge_coordinates = edge_coordinates

    # saving
    write_swc = _write_swc_func

