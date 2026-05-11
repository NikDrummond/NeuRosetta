"""Functions for handling / finding subtrees"""

from numpy import ndarray
from graph_tool.all import GraphView

from ...core import _Tree
from ...utils.graph_utils import (
    bfsearch,
    SubtreeMaskVisitor,
    subgraph_score,
    max_subgraph_ind,
    extract_subgraph,
    partition_asymmetry,
)
from .tree_checks import tree_has_property
from .path_lengths import euclidean_edge_length


def mask_subtree_from_root(tree: _Tree, root: int, bind: bool = True):
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
    vis = bfsearch(
        g=tree.graph,
        visitor=SubtreeMaskVisitor,
        init_vertex_properties={"v_subtree_mask": "bool"},
        init_edge_properties={"e_subtree_mask": "bool"},
        root=root,
        bind=bind,
    )

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
        euclidean_edge_length(tree, bind=True)

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
        score_subtrees(tree, bind=True)

    return max_subgraph_ind(tree.graph)


def extract_subtree(
    tree: _Tree, revert_properties: bool = True
) -> None:
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    revert_properties : bool, optional
        _description_, by default True

    Returns
    -------
    None | Graph
        _description_
    """

    # make sure we have a mask
    if not tree_has_property(tree, "v_subtree_mask", "v"):
        mask_subtree_from_root(tree, max_subtree_ind(tree), bind=True)

    extract_subgraph(tree.graph, revert_properties = revert_properties)

def node_partition_asymmetry(tree: _Tree, weighted: bool = True, bind: bool = False) -> None | ndarray:
    """
    Compute partition asymmetry at each branching node in a directed tree graph.

    For a binary branching node with downstream leaf counts r and s (one per child
    subtree), the unweighted partition asymmetry is:

        PA = |r - s| / (r + s - 1)

    For non-binary nodes the score is the mean over all C(k, 2) child-pair
    combinations, where each pair is evaluated with the formula above.

    In the weighted variant, each node's score is additionally multiplied by the
    fraction of total cable length contained in its subtree:

        PA_weighted = PA * (subtree_cable[v] / total_cable)

    Parameters
    ----------
    tree : Tree
        Neuron Tree graph
    weighted : bool, optional
        If True compute the cable-weighted version.  Requires the "Path_length"
        edge property.  By default False.
    bind : bool, optional
        If True attach the result as vertex property ``g.vp["partition_asymmetry"]``
        and return None.  If False return the raw ndarray.  By default True.

    Returns
    -------
    None | ndarray
        Array of partition-asymmetry scores, one per vertex.  Leaf nodes and the
        root receive a score of 0.  Non-NaN values exist only for branching nodes
        (out-degree >= 2).
    """
    if bind:
        partition_asymmetry(tree.graph, weighted = weighted, bind = bind)
        return
    return partition_asymmetry(tree.graph, weighted = weighted, bind = bind)
