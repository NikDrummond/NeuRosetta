"""Mask and extract a subtree, including attached synapses."""

from __future__ import annotations

from numpy import flatnonzero

from ...core import _Tree
from ...utils.graph_utils import SubtreeMaskVisitor, bfsearch, extract_subgraph
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import has_property
from .tree_subtree_scores import get_max_subtree_node


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


def get_subtree(
    tree: _Tree,
    revert_properties: bool = True,
    *,
    synapses: str = "mapped_only",
) -> None:
    """Extract the masked subtree from the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with vertex and edge subtree masks.
    revert_properties : bool, optional
        If True, remove non-core properties after extraction. By default True.
    synapses : {"mapped_only", "all", "none"}, optional
        How to retain attached synapses after extraction:

        - ``"mapped_only"`` (default): keep synapses with ``mapped=True`` on
          retained edges (remapped to the new edge index space). Unmapped
          tables are cleared — geometry membership is unknown.
        - ``"all"``: keep any synapse whose nearest edge was retained
          (including ``mapped=False``); if the table was never mapped, keep all
          rows and invalidate mapping.
        - ``"none"``: drop all synapses.

    Returns
    -------
    None
    """
    from .tree_synapses import (
        clear_synapses,
        filter_tree_synapses_to_edges,
        get_synapses,
        invalidate_synapse_mapping,
    )

    if synapses not in ("mapped_only", "all", "none"):
        raise ValueError(f"synapses={synapses!r} invalid; expected 'mapped_only', 'all', or 'none'")

    retained_edges = None
    syn = get_synapses(tree)
    if (
        synapses != "none"
        and syn is not None
        and syn.is_mapped
        and has_property(tree, "e_subtree_mask", "e")
    ):
        mask = tree.graph.ep["e_subtree_mask"].a.astype(bool)
        retained_edges = flatnonzero(mask)

    if not has_property(tree, "v_subtree_mask", "v"):
        mask_subtree_from_root(tree, get_max_subtree_node(tree), bind=True)
        if synapses != "none" and syn is not None and syn.is_mapped and retained_edges is None:
            mask = tree.graph.ep["e_subtree_mask"].a.astype(bool)
            retained_edges = flatnonzero(mask)

    extract_subgraph(tree.graph, revert_properties=revert_properties)

    if synapses == "none":
        if syn is not None:
            clear_synapses(tree)
        return

    if retained_edges is not None:
        filter_tree_synapses_to_edges(tree, retained_edges, mapped_only=(synapses == "mapped_only"))
    elif syn is not None:
        if synapses == "mapped_only":
            clear_synapses(tree)
        else:
            invalidate_synapse_mapping(tree)


enrich_tree_graph_docstrings(globals())
