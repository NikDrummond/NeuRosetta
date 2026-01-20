from graph_tool.all import Graph

from ..core import _Tree
from ..tree_graphs.vertex_inds import get_root, get_leaves, get_branches, get_core_inds, get_edges
from ..tree_graphs.counting import count_roots, count_edges, count_branches, count_leaves, count_vertices
from ..tree_graphs.coordinates import vertex_coordinates, edge_coordinates
from ..plotting.plot_2d import plot_2d
from ..plotting.plot_3d import plot_3d

from ..io_utils.swc_utils import write_swc as _write_swc_func
from ..io_utils.nr_utils import save as _save


class Tree(_Tree):

    def __init__(self, ID:int, metadata:dict, graph:Graph) -> None:
        super().__init__(ID = ID, metadata = metadata, graph = graph)

    ### get indices 
    root_index = get_root
    leaf_indices = get_leaves
    branch_indices = get_branches
    core_indices = get_core_inds
    edge_indices = get_edges

    ### counting
    num_roots = count_roots
    num_nodes = count_vertices
    num_edges = count_edges
    num_branches = count_branches
    num_leaves = count_leaves

    ### coordinates
    get_node_coordinates = vertex_coordinates
    get_edge_coordinates = edge_coordinates

    # saving
    write_swc = _write_swc_func
    save = _save

    # plotting
    show_2d = plot_2d
    show_3d = plot_3d

