from numpy import concatenate, vstack
from graph_tool.all import BFSVisitor, DFSVisitor, Graph

from ..graphs.traversals import (
    _BF_search,
    Tree_depth,
    _DF_search,
    PostOrderVisitor,
    ReduceVisitor,
)
from ..core import _Tree

### Generic BF Search


def BF_search(
    tree: _Tree,
    visitor: BFSVisitor,
    init_kwargs:dict = {},
    init_properties: dict = {},
    root: int | None = None,
    bind: bool = True,
):
    if root is None:
        root = tree.root_index()

    return _BF_search(g = tree.graph, visitor = visitor, init_kwargs = init_kwargs, init_properties = init_properties, root = root, bind = bind)


## Generic DF Search


def DF_search(
    tree: _Tree,
    visitor: DFSVisitor,
    init_kwargs: dict = {},
    init_properties: dict = {},
    root: int | None = None,
    bind: bool = True,
):
    if root is None:
        root = tree.root_index()

    return _DF_search(g = tree.graph, visitor = visitor, init_kwargs = init_kwargs, init_properties = init_properties, root = root, bind = bind)


### Specific BF Searches


# depths
def compute_depths(tree: _Tree, root: int | None = 0, bind: bool = True):
    """get node/edge depth from root"""

    if root is None:
        root = tree.root_index()

    out = BF_search(tree = tree, visitor = Tree_depth, init_properties = {"depth": "int"}, root = root, bind=bind)

    return out["depth"].a if not bind else None


### Specific DF searches


# post-order taversal
def compute_post_order(tree: _Tree, root: int | None = None, bind: bool = True):
    """get post-order traversal of tree"""

    if root is None:
        root = tree.root_index()

    vis = DF_search(tree= tree, visitor = PostOrderVisitor, root = root)

    if bind:
        tree.graph.vertex_properties["post_order"] = tree.graph.new_vp(
            "int", vis.post_order
        )
    else:
        return vis


def reduce_graph(tree: _Tree) -> Graph:
    """Generate a reduced version of the given tree

    Parameters
    ----------
    tree : _Tree
        _description_
    starts : ndarray
        _description_
    stops : ndarray
        _description_
    root : int | None, optional
        _description_, by default 0

    Returns
    -------
    DFSVisitor
        _description_
    """
    # set root

    # check that we have path lengths
    if not tree.check_property("Path_length"):
        tree.get_edge_lengths(bind=True)

    root = tree.root_index()

    ### stops are all leaves and branches excluding the root
    stops = tree.core_indices(include_root=False)
    # starts are all branches including the root
    starts = tree.branch_indices()
    # make sure we have the root
    if tree.root_index() not in starts:
        starts = concatenate([starts, [tree.root_index()]])
    # generate visitor
    vis = _DF_search(
        tree.graph,
        ReduceVisitor,
        {"graph": tree.graph, "starts": starts, "stops": stops},
        root=root,
        bind=False,
    )

    # create edge list
    edges = vstack((vis.edge_source, vis.edge_target)).T
    # make graph from edge list
    g = Graph(edges, hashed=True, hash_type=int)
    # add path lengths to edges
    g.ep["Path_length"] = g.new_ep("float", vis.path_lengths)
    # add coordinates to verts
    coords = tree.get_node_coordinates()[g.vp["ids"].a]
    g.vp["coordinates"] = g.new_vp("vector<double>", coords)
    # add ID (from original)
    g.gp["ID"] = g.new_gp("long", tree.ID)
    # add metadata and update reduced
    meta = tree.graph.gp["metadata"].copy()
    meta["file_path"] = ""
    meta["isReduced"] = True
    g.gp["metadata"] = g.new_gp("object", meta)
    # radius
    g.vp["radius"] = g.new_vp("double", tree.graph.vp["radius"].a[g.vp["ids"].a])
    # node type
    g.vp["node_type"] = g.new_vp("int", tree.graph.vp["node_type"].a[g.vp["ids"].a])

    return g
