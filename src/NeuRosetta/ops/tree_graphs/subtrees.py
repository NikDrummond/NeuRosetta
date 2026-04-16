""" Functions for handling / finding subtrees """

from ...core import _Tree
from ...utils.graph_utils import (
    bfsearch,
    SubtreeMaskVisitor,
)

def mask_subtree_from_root(tree:_Tree, root: int, bind:bool = True):
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    root : int
        _description_
    bind : bool, optional
        _description_, by default True
    """
    vis = bfsearch(g = tree.graph, 
                visitor = SubtreeMaskVisitor, 
                init_vertex_properties={"v_subtree_mask":"bool"},
                init_edge_properties = {"e_subtree_mask":"bool"},
                root = root,
                bind = bind)

    return vis if not bind else None
