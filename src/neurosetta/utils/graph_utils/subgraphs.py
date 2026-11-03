"""Functions for handling subgraphs and computing subgraph metrics."""

from __future__ import annotations

from itertools import combinations
from typing import Literal


from graph_tool.all import Graph
from graph_tool.topology import topological_sort
from numpy import (
    argmax,
    nan,
    nan_to_num,
    ndarray,
    where,
    zeros,
    any as _any,
    flatnonzero,
    asarray,
    isfinite,
    all as _all,
    mean,
)

from .gt_properties import (
    g_has_property,
    raise_internal_property_missing,
    revert_core_properties,
)


def _cable_length_property(g: Graph):
    """Return the edge property map used as physical cable length.

    Prefers ``Path_length``; falls back to ``Euclidean_length`` (reduced trees).
    """
    if g_has_property(g, "Path_length", "e"):
        return g.ep["Path_length"]
    if g_has_property(g, "Euclidean_length", "e"):
        return g.ep["Euclidean_length"]
    raise_internal_property_missing(g, "Path_length", "e")


def subgraph_score(g: Graph, bind: bool = True) -> None | ndarray:
    """Compute subgraph score for each vertex in a directed tree.

    The score combines two terms for branch points (out-degree >= 2):
    - Cable fraction: 1 - (subtree_cable / total_cable)
    - Leaf fraction: subtree_leaves / total_leaves

    Non-branch vertices receive a score of 0.

    Parameters
    ----------
    g : Graph
        Directed tree graph with edge property ``Path_length`` or
        ``Euclidean_length``.
    bind : bool, optional
        If True, bind the result as vertex property "subgraph_score" and return
        None. If False, return the score array. By default True.

    Returns
    -------
    None | ndarray
        If bind=False, returns array of scores with shape (n_vertices,).
        Otherwise returns None and adds "subgraph_score" vertex property.
    """
    # globals
    path_length = _cable_length_property(g)

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
    """Return the vertex index with the maximum subgraph score.

    Parameters
    ----------
    g : Graph
        Directed tree graph. If "subgraph_score" vertex property exists, it is
        used. Otherwise, the score is computed internally.

    Returns
    -------
    int
        Vertex index with the maximum subgraph score.
    """
    if g_has_property(g, "subgraph_score", "v"):
        return argmax(g.vp["subgraph_score"].a)

    return argmax(subgraph_score(g, bind=False))


def extract_subgraph(g: Graph, revert_properties: bool = True) -> Graph:
    """Extract a subgraph using vertex and edge filters.

    Requires the graph to have "v_subtree_mask" (vertex) and "e_subtree_mask"
    (edge) properties set as boolean filters.

    Parameters
    ----------
    g : Graph
        Graph with vertex filter "v_subtree_mask" and edge filter "e_subtree_mask".
    revert_properties : bool, optional
        If True, remove non-core properties after extraction. By default True.

    Returns
    -------
    Graph
        The filtered and purged graph (modified in place and returned).
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

    return g


def partition_asymmetry(
    g: Graph,
    mode: Literal["unweighted", "weighted", "van_pelt", "cable"] = "weighted",
    bind: bool = True,
    *,
    include_root: bool = True,
    property_name: str | None = None,
) -> None | ndarray:
    r"""Compute partition asymmetry at branching nodes of a directed tree.

    Supported modes
    ---------------
    unweighted:
        |r - s| / (r + s - 1)

    weighted:
        Unweighted partition asymmetry multiplied by the fraction
        of total dendritic cable contained in the branching node's
        downstream subtree.

    van_pelt:
        |r - s| / (r + s - 2)
        The (1, 1) partition is assigned zero.

    cable:
        |C1 - C2| / (C1 + C2)
        C1 and C2 include each daughter's connecting edge and
        all downstream cable.

    For multifurcations, pairwise asymmetry is averaged over all
    unordered pairs of daughter subtrees.

    Parameters
    ----------
    g : Graph
        Directed graph-tool tree, oriented from root to leaves.
    mode : str, default "weighted"
        Asymmetry definition.
    bind : bool, default True
        If True, store results as a vertex property and return None.
        Otherwise, return an ndarray.
    include_root : bool, default True
        Include the root when it has at least two children.
    property_name : str or None, default None
        Vertex property name. Defaults to
        "partition_asymmetry_<mode>", with "van_pelt"
        normalized to "vanpelt".

    Returns
    -------
    None | ndarray
        One score per vertex. Non-branching nodes receive zero.

    Notes
    -----
    Cable-based modes require the "Path_length" edge property
    containing true dendritic cable lengths.

    The input must be a connected, directed, acyclic rooted tree.
    """

    valid_modes = {"unweighted", "weighted", "van_pelt", "cable"}

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown partition asymmetry mode: {mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    if property_name is None:
        suffix = mode.replace("van_pelt", "vanpelt")
        property_name = f"partition_asymmetry_{suffix}"

    needs_cable = mode in {"weighted", "cable"}

    n = g.num_vertices()

    if n == 0:
        raise ValueError("Cannot calculate partition asymmetry on an empty graph.")

    vertices = g.get_vertices()
    out_degrees = g.get_out_degrees(vertices)
    in_degrees = g.get_in_degrees(vertices)

    roots = vertices[in_degrees == 0]

    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}.")

    root = int(roots[0])

    # ------------------------------------------------------------
    # Edge data
    # ------------------------------------------------------------

    if needs_cable:
        if "Path_length" not in g.ep:
            raise ValueError(
                f"Mode {mode!r} requires the 'Path_length' edge "
                "property containing true dendritic cable lengths."
            )

        path_length = g.ep["Path_length"]
        edges = g.get_edges([path_length])

        edge_lengths = edges[:, 2].astype(float)

        if not _all(isfinite(edge_lengths)):
            raise ValueError("Path_length contains non-finite values.")

        if _any(edge_lengths <= 0):
            raise ValueError("Path_length must contain positive cable lengths.")

        total_cable = float(edge_lengths.sum())

        if total_cable <= 0:
            raise ValueError("Total dendritic cable length must be positive.")

    else:
        edges = g.get_edges()
        edge_lengths = None
        total_cable = None

    # ------------------------------------------------------------
    # Build child adjacency
    # ------------------------------------------------------------

    children = [[] for _ in range(n)]
    daughter_edge_lengths = [[] for _ in range(n)]

    for edge_idx, edge in enumerate(edges):
        src = int(edge[0])
        tgt = int(edge[1])

        children[src].append(tgt)

        if needs_cable:
            daughter_edge_lengths[src].append(float(edge_lengths[edge_idx]))

    # ------------------------------------------------------------
    # Bottom-up subtree accumulation
    # ------------------------------------------------------------

    subtree_leaves = (out_degrees == 0).astype(float)
    subtree_cable = zeros(n, dtype=float)

    topo = asarray(topological_sort(g), dtype=int)

    if len(topo) != n:
        raise ValueError(
            "Topological traversal did not cover all vertices. "
            "Check that the graph is a valid directed tree."
        )

    for v in reversed(topo):
        if not children[v]:
            continue

        child_ids = asarray(children[v], dtype=int)

        subtree_leaves[v] += subtree_leaves[child_ids].sum()

        if needs_cable:
            subtree_cable[v] = (
                sum(daughter_edge_lengths[v]) + subtree_cable[child_ids].sum()
            )

    # ------------------------------------------------------------
    # Partition asymmetry
    # ------------------------------------------------------------

    score = zeros(n, dtype=float)

    for v in topo:
        if out_degrees[v] < 2:
            continue

        if not include_root and v == root:
            continue

        child_ids = children[v]
        child_leaves = subtree_leaves[child_ids]

        # Complete daughter cable includes connecting edges.
        if needs_cable:
            child_cable = subtree_cable[child_ids] + asarray(
                daughter_edge_lengths[v], dtype=float
            )

        pair_scores = []

        for i, j in combinations(range(len(child_ids)), 2):

            r = float(child_leaves[i])
            s = float(child_leaves[j])

            if mode in {"unweighted", "weighted"}:

                denom = r + s - 1.0

                pair_pa = abs(r - s) / denom if denom > 0 else 0.0

            elif mode == "van_pelt":

                denom = r + s - 2.0

                pair_pa = abs(r - s) / denom if denom > 0 else 0.0

            elif mode == "cable":

                c1 = float(child_cable[i])
                c2 = float(child_cable[j])

                denom = c1 + c2

                pair_pa = abs(c1 - c2) / denom if denom > 0 else 0.0

            pair_scores.append(pair_pa)

        pa = float(mean(pair_scores))

        # Existing NeuRosetta cable-weighted definition:
        # weight the node-level partition asymmetry by the
        # fraction of total cable downstream of this node.
        if mode == "weighted":
            pa *= subtree_cable[v] / total_cable

        score[v] = pa

    # ------------------------------------------------------------
    # Return or bind
    # ------------------------------------------------------------

    if bind:
        g.vp[property_name] = g.new_vp("double", vals=score)
        return None

    return score
