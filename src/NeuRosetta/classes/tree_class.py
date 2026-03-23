from graph_tool.all import Graph

from ..core import _Tree
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

from ..plotting.plot_2d import plot_2d
from ..plotting.plot_3d import plot_3d
from ..plotting.plot_dendrogram import plot_dendrogram

from ..io_utils.swc_utils import export_swc as _write_swc_func
from ..io_utils.nr_utils import save as _save


class Tree(_Tree):

    def __init__(self, ID: int, metadata: dict, graph: Graph) -> None:
        super().__init__(ID=ID, metadata=metadata, graph=graph)

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

    ### degrees
    get_degree_array = get_degrees
    get_degree_distribution = degree_distribution

    ### Traversals
    Breadth_first_search = BF_search
    Depth_first_search = DF_search
    Get_post_order_traversal = compute_post_order

    ### Topological bits
    Get_node_depths = compute_depths

    # saving
    export_swc = _write_swc_func
    save = _save

    # plotting
    show_2d = plot_2d
    show_3d = plot_3d
    show_dendrogram = plot_dendrogram

    # checks
    is_reduced = is_Reduced
    update_reduced = update_reduced
    check_property = has_property
