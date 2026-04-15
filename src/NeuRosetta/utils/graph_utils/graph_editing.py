"""Functions to modify structure of graphs"""

from numpy import concatenate, vstack
from graph_tool.all import Graph, GraphView

from .gt_properties import raise_internal_property_missing
from .traversals import dfsearch, ReduceVisitor, bf_iterator
from .vertex_inds import root_index, core_indices, branch_indices
from .coordinates import vertex_coordinates


### reduce
def reduce_graph(g: Graph) -> Graph:
    """_summary_

    Parameters
    ----------
    g : Graph
        Graph object

    Returns
    -------
    Graph
        _description_
    """

    # we need path a bunch of properties
    raise_internal_property_missing(g, "metadata", "g")
    raise_internal_property_missing(g, "ID", "g")
    raise_internal_property_missing(g, "coordinates", "v")
    raise_internal_property_missing(g, "radius", "v")
    raise_internal_property_missing(g, "node_type", "v")
    raise_internal_property_missing(g, "Path_length", "e")

    # get root
    root = root_index(g)

    ### stops are all leaves and branches excluding the root
    stops = core_indices(g, include_root=False)
    # starts are all branches including the root
    starts = branch_indices(g)
    # make sure we have the root
    if root not in starts:
        starts = concatenate([starts, [root]])

    # Traversal
    # generate visitor
    vis = dfsearch(
        g,
        ReduceVisitor,
        {"graph": g, "starts": starts, "stops": stops},
        root=root,
        bind=False,
    )

    # create edge list
    edges = vstack((vis.edge_source, vis.edge_target)).T
    # make graph from edge list
    g_red = Graph(edges, hashed=True, hash_type=int)
    # add path lengths to edges
    g_red.ep["Path_length"] = g_red.new_ep("float", vis.path_lengths)
    # add coordinates to verts
    coords = vertex_coordinates(g)[g_red.vp["ids"].a]
    g_red.vp["coordinates"] = g_red.new_vp("vector<double>", coords)
    # add ID (from original)
    g_red.gp["ID"] = g_red.new_gp("long", g.gp["ID"])
    # add metadata and update reduced
    meta = g.gp["metadata"].copy()
    meta["file_path"] = ""
    meta["isReduced"] = True
    g_red.gp["metadata"] = g_red.new_gp("object", meta)
    # radius
    g_red.vp["radius"] = g_red.new_vp("double", g.vp["radius"].a[g_red.vp["ids"].a])
    # node type
    g_red.vp["node_type"] = g_red.new_vp("int", g.vp["node_type"].a[g_red.vp["ids"].a])

    return g_red

### reroot


def reroot_graph(g: Graph, root: int) -> None:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_
    root : int
        _description_
    """

    # we need path a bunch of properties
    raise_internal_property_missing(g, "metadata", "g")
    raise_internal_property_missing(g, "ID", "g")
    raise_internal_property_missing(g, "coordinates", "v")
    raise_internal_property_missing(g, "radius", "v")
    raise_internal_property_missing(g, "node_type", "v")

    # Undirected Graphview to get edge list from new root
    g_view = GraphView(g, directed=False)
    # edge list from new root (BFS)
    edges = bf_iterator(g_view, root, array=True)
    # new graph
    g_new = Graph(edges, hashed=True, hash_type="int")

    ### migrate properties

    # ID
    g_new.gp["ID"] = g_new.new_gp("long", g_view.gp["ID"])
    # metadata
    g_new.gp["metadata"] = g_new.new_gp("object", g_view.gp["metadata"])
    # coordinates
    coords = g_view.vp["coordinates"].get_2d_array().T[g_new.vp["ids"].a]
    g_new.vp["coordinates"] = g_new.new_vp("vector<double>", coords)
    # node_type
    g_new.vp["node_type"] = g_new.new_vp(
        "int", g_view.vp["node_type"].a[g_new.vp["ids"].a]
    )
    # radius
    g_new.vp["radius"] = g_new.new_vp(
        "double", g_view.vp["radius"].a[g_new.vp["ids"].a]
    )

    return g_new
