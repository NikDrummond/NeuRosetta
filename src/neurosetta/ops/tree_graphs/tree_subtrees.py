"""Functions for handling and finding subtrees."""

from numpy import ndarray

# from graph_tool.all import GraphView
from ...core import _Tree
from ...utils.graph_utils import (
    SubtreeMaskVisitor,
    bfsearch,
    extract_subgraph,
    max_subgraph_ind,
    partition_asymmetry,
    subgraph_score,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import has_property
from .tree_path_lengths import get_edge_length


def mask_subtree_from_root(tree: _Tree, root: int, bind: bool = True) -> SubtreeMaskVisitor | None:
    """Create vertex and edge masks for a subtree rooted at the given vertex.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Root vertex index of the subtree.
    bind : bool, optional
        If True, bind the masks as vertex/edge properties to the graph and return
        None. If False, return the SubtreeMaskVisitor. By default True.

    Returns
    -------
    SubtreeMaskVisitor | None
        Visitor with mask properties if bind=False, otherwise None.
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


def get_subtree_scores(tree: _Tree, bind: bool = True) -> ndarray | None:
    """Compute subgraph scores for all vertices in the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, bind the scores as a vertex property and return None.
        If False, return the score array. By default True.

    Returns
    -------
    ndarray | None
        Array of subgraph scores if bind=False, otherwise None.
    """
    if not has_property(tree, "Path_length", "e"):
        get_edge_length(tree, bind=True)

    score = subgraph_score(tree.graph, bind)

    return score if not bind else None


def get_max_subtree_node(tree: _Tree) -> int:
    """Get the vertex index with the maximum subgraph score.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Vertex index with the maximum subgraph score.
    """
    # make sure we have Path_length
    if not has_property(tree, "subgraph_score", "v"):
        get_subtree_scores(tree, bind=True)

    return max_subgraph_ind(tree.graph)


def get_subtree(tree: _Tree, revert_properties: bool = True) -> None:
    """Extract the masked subtree from the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with vertex and edge subtree masks.
    revert_properties : bool, optional
        If True, remove non-core properties after extraction. By default True.

    Returns
    -------
    None
    """
    # make sure we have a mask
    if not has_property(tree, "v_subtree_mask", "v"):
        mask_subtree_from_root(tree, get_max_subtree_node(tree), bind=True)

    extract_subgraph(tree.graph, revert_properties=revert_properties)


def get_partition_asymmetry(
    tree: _Tree, weighted: bool = True, bind: bool = False
) -> ndarray | None:
    r"""Compute partition asymmetry at each branching node in a directed tree graph.

    For a binary branching node with downstream leaf counts r and s (one per child
    subtree), the unweighted partition asymmetry is:

        PA = \|r - s\| / (r + s - 1)

    For non-binary nodes the score is the mean over all C(k, 2) child-pair
    combinations, where each pair is evaluated with the formula above.

    In the weighted variant, each node's score is additionally multiplied by the
    fraction of total cable length contained in its subtree:

        PA_weighted = PA * (subtree_cable[v] / total_cable)

    Parameters
    ----------
    tree : _Tree
        Neuron tree graph.
    weighted : bool, optional
        If True compute the cable-weighted version. Requires the "Path_length"
        edge property. By default True.
    bind : bool, optional
        If True attach the result as vertex property
        ``g.vp["partition_asymmetry"]`` and return None. If False return the
        raw ndarray. By default False.

    Returns
    -------
    ndarray | None
        Array of partition-asymmetry scores, one per vertex. Leaf nodes and the
        root receive a score of 0. Non-zero values exist only for branching nodes
        (out-degree >= 2).
    """
    if bind:
        partition_asymmetry(tree.graph, weighted=weighted, bind=bind)
        return
    return partition_asymmetry(tree.graph, weighted=weighted, bind=bind)


enrich_tree_graph_docstrings(globals())
