"""Functions for handling subgraphs"""

from numpy import (
    zeros,
    where,
    nan,
    nan_to_num,
    argmax,
    ndarray,
)
from itertools import combinations

from graph_tool.all import Graph
from graph_tool.topology import topological_sort

from .gt_properties import (
    raise_internal_property_missing,
    g_has_property,
    revert_core_properties,
)


def subgraph_score(g: Graph, bind: bool = True) -> None | ndarray:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    None | ndarray
        _description_
    """

    # make sure we have path length
    raise_internal_property_missing(g, "Path_length", "e")

    # globals
    path_length = g.ep["Path_length"]

    total_cable = path_length.a.sum()
    out_degrees = g.get_out_degrees(g.get_vertices())
    is_leaf = out_degrees == 0
    total_leaves = is_leaf.sum()

    # Edge lookup arrays
    edges = g.get_edges([path_length])
    e_src = edges[:, 0].astype(int)
    e_tgt = edges[:, 1].astype(int)
    e_pl = edges[:, 2]

    # subtree accumulators
    n = g.num_vertices()
    subtree_cable = zeros(n)
    subtree_leaves = is_leaf.astype(float)

    # reverse topological pass
    topo = topological_sort(g)

    for v in reversed(topo):
        mask = e_src == v
        children = e_tgt[mask]
        if children.size == 0:
            continue
        subtree_cable[v] = e_pl[mask].sum() + subtree_cable[children].sum()
        subtree_leaves[v] += subtree_leaves[children].sum()

    # score for branches
    branch_mask = out_degrees >= 2

    score = where(
        branch_mask,
        (1.0 - (subtree_cable / total_cable)) + (subtree_leaves / total_leaves),
        nan,
    )
    # set nans (non-branches) to 0
    score = nan_to_num(score, nan=0.0)

    if bind:
        g.vp["subgraph_score"] = g.new_vp("double", score)
        return
    return score


def max_subgraph_ind(g: Graph) -> int:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_

    Returns
    -------
    int
        _description_
    """

    if g_has_property(g, "subgraph_score", "v"):
        return argmax(g.vp["subgraph_score"].a)

    return argmax(subgraph_score(g, bind=False))


def extract_subgraph(g: Graph, revert_properties: bool = True) -> Graph:
    """_summary_

    Parameters
    ----------
    g : Graph
        _description_

    Returns
    -------
    Graph
        _description_
    """

    # make sure we have needed masks
    raise_internal_property_missing(g, "v_subtree_mask", "v")
    raise_internal_property_missing(g, "e_subtree_mask", "e")

    g.set_vertex_filter(g.vp["v_subtree_mask"])
    g.set_edge_filter(g.ep["e_subtree_mask"])

    # purge
    g.purge_vertices()
    g.purge_edges()

    if revert_properties:
        revert_core_properties(g)


def partition_asymmetry(
    g: Graph, weighted: bool = False, bind: bool = True
) -> None | ndarray:
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
    g : Graph
        A graph-tool Graph object representing a directed tree.  Must carry an
        edge property "Path_length" when ``weighted=True``.
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

    if weighted:
        raise_internal_property_missing(g, "Path_length", "e")

    n = g.num_vertices()
    out_degrees = g.get_out_degrees(g.get_vertices())
    is_leaf = out_degrees == 0

    # Edge lookups
    if weighted:
        path_length = g.ep["Path_length"]
        edges = g.get_edges([path_length])
        e_pl = edges[:, 2]
        total_cable = e_pl.sum()
    else:
        edges = g.get_edges()
        e_pl = None
        total_cable = None

    e_src = edges[:, 0].astype(int)
    e_tgt = edges[:, 1].astype(int)

    # Topological pass – accumulate subtree leaf-counts (and cable)       #
    subtree_leaves = is_leaf.astype(float).copy()
    subtree_cable = zeros(n)

    topo = topological_sort(g)

    for v in reversed(topo):
        mask = e_src == v
        children = e_tgt[mask]
        if children.size == 0:
            continue
        subtree_leaves[v] += subtree_leaves[children].sum()
        if weighted:
            subtree_cable[v] = e_pl[mask].sum() + subtree_cable[children].sum()

    # PA per branch
    score = zeros(n)

    for v in reversed(topo):
        if out_degrees[v] < 2:  # leaves and single-child nodes → 0
            continue

        mask = e_src == v
        children = e_tgt[mask]

        # leaf counts of each child's subtree
        child_leaves = subtree_leaves[children]  # shape (k,)

        # mean PA over all C(k, 2) pairs
        pair_scores = []
        for i, j in combinations(range(len(children)), 2):
            r, s = child_leaves[i], child_leaves[j]
            denom = r + s - 1.0
            if denom <= 0:
                # degenerate case: both subtrees have a single shared leaf - this should be impossible
                pair_scores.append(0.0)
            else:
                pair_scores.append(abs(r - s) / denom)

        pa = sum(pair_scores) / len(pair_scores)

        if weighted:
            pa *= subtree_cable[v] / total_cable

        score[v] = pa

    if bind:
        g.vp["partition_asymmetry"] = g.new_vp("double", score)
        return
    return score
