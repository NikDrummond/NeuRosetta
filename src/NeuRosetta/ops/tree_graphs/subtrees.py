""" Functions for handling / finding subtrees """
from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    bfsearch,
    SubtreeMaskVisitor,
    subgraph_score,
    max_subgraph_ind
)
from .tree_checks import tree_has_property
from .path_lengths import euclidean_edge_length

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

def score_subtrees(tree: _Tree, bind: bool = True) -> None | ndarray:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    None | ndarray
        _description_
    """

    if not tree_has_property(tree, "Path_length", "e"):
        euclidean_edge_length(tree, bind = True)

    score = subgraph_score(tree.graph, bind)

    return score if not bind else None

def max_subtree_ind(tree: _Tree) -> int:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_

    Returns
    -------
    int
        _description_
    """

    # make sure we have Path_length
    if not tree_has_property(tree, "subgraph_score", "v"):
        score_subtrees(tree, bind = True)

    return max_subgraph_ind(tree.graph)
