"""Functions for handling subgraphs"""

from numpy import (
    zeros,
    where,
    nan,
    nan_to_num,
    argmax,
    ndarray,
)

from graph_tool.all import Graph
from graph_tool.topology import topological_sort

from .gt_properties import raise_internal_property_missing, g_has_property


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
