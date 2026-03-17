from graph_tool all import BFSVisitor, DFSVisitor

from ..graphs.traversals import _BF_search, Tree_depth, _DF_search, PostOrderVisitor
from ..core import _Tree

### Generic BF Search

def BF_search(tree: _Tree, visitor: BFSVisitor, init_properties: dict {}, root: int | None = None, bind: bool = True):
    if root is None:
        root = tree.root_index()

    return _BF_search(tree.graph, visitor, init_properties, root, bind)

## Generic DF Search

def DF_search(tree: _Tree, visitor: DFSVisitor, init_properties: dict {}, root: int | None = None, bind: bool = True):
    if root is None:
        root = tree.root_index()

    return _DF_search(tree.graph, visitor, init_properties, root, bind)

### Specific BF Searches

# depths
def compute_depths(tree: _Tree, root: int | None = 0, bind: bool = True):
    """ get node/edge depth from root"""

    if root is None:
        root = tree.root_index()

    out = BF_search(tree.graph, Tree_depth, {"depth": "int"}, root, bind=bind)

    return out["depth"] if not bind else None


### Specific DF searches

# post-order taversal
def compute_post_order(tree: _Tree, root: int | None = 0, bind: bool = True):
    """ get post-order traversal of tree"""

    if root is None:
        root = tree.root_index()

    vis = DF_search(tree.graph, PostOrderVisitor, {}, root)

    if bind:
        tree.graph.vertex_properties["post_order"] = tree.graph.new_vp('int', vis.post_order)
    else:
        return vis