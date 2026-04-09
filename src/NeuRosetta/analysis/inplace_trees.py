from graph_tool.all import GraphView, bfs_iterator, Graph

from NeuRosetta.utils.graph_utils.properties import g_has_property
from NeuRosetta.ops.tree_graphs.path_lengths import euclidean_edge_length

from NeuRosetta.core import _Tree

def _reroot_tree(tree: _Tree, root: int) -> None:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    root : int
        _description_
    """

    # Undirected Graphview to get edge list from new root
    g_view = GraphView(tree.graph, directed = False)
    # edge list from new root (BFS)
    edges = bfs_iterator(g_view, source = root, array = True)
    # new graph
    g = Graph(edges, hashed = True, hash_type = "int")


    ### migrate properties

    # ID
    g.gp['ID'] = g.new_gp('long',g_view.gp['ID'])
    # metadata
    g.gp['metadata'] = g.new_gp('object', g_view.gp['metadata'])
    # coordinates
    coords = g_view.vp['coordinates'].get_2d_array().T[g.vp['ids'].a]
    g.vp['coordinates'] = g.new_vp('vector<double>', coords)
    # node_type
    g.vp['node_type'] = g.new_vp('int', g_view.vp['node_type'].a[g.vp['ids'].a])
    # radius
    g.vp['radius'] = g.new_vp('double', g_view.vp['radius'].a[g.vp['ids'].a])

    # update graph in tree
    tree.graph = g

    ### optionally add path length
    if g_has_property(g_view, 'Path_length', 'e'):
        # add lengths
        euclidean_edge_length(tree, bind = True)
