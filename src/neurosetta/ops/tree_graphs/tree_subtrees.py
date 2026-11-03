"""Functions for handling and finding subtrees."""

from typing import Literal
from numpy import flatnonzero, ndarray

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


def mask_subtree_from_root(
    tree: _Tree, root: int, bind: bool = True
) -> SubtreeMaskVisitor | None:
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
    if not (
        has_property(tree, "Path_length", "e")
        or has_property(tree, "Euclidean_length", "e")
    ):
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
    synapses : {\"mapped_only\", \"all\", \"none\"}, optional
        How to retain attached synapses after extraction:

        - ``\"mapped_only\"`` (default): keep synapses with ``mapped=True`` on
          retained edges (remapped to the new edge index space). Unmapped
          tables are cleared — geometry membership is unknown.
        - ``\"all\"``: keep any synapse whose nearest edge was retained
          (including ``mapped=False``); if the table was never mapped, keep all
          rows and invalidate mapping.
        - ``\"none\"``: drop all synapses.

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
        raise ValueError(
            f"synapses={synapses!r} invalid; expected 'mapped_only', 'all', or 'none'"
        )

    # Capture retained edge indices before purge remaps them.
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

    # make sure we have a mask
    if not has_property(tree, "v_subtree_mask", "v"):
        mask_subtree_from_root(tree, get_max_subtree_node(tree), bind=True)
        if (
            synapses != "none"
            and syn is not None
            and syn.is_mapped
            and retained_edges is None
        ):
            mask = tree.graph.ep["e_subtree_mask"].a.astype(bool)
            retained_edges = flatnonzero(mask)

    extract_subgraph(tree.graph, revert_properties=revert_properties)

    if synapses == "none":
        if syn is not None:
            clear_synapses(tree)
        return

    if retained_edges is not None:
        filter_tree_synapses_to_edges(
            tree, retained_edges, mapped_only=(synapses == "mapped_only")
        )
    elif syn is not None:
        if synapses == "mapped_only":
            clear_synapses(tree)
        else:
            invalidate_synapse_mapping(tree)


def get_partition_asymmetry(
    tree: _Tree,
    mode: Literal["unweighted", "weighted", "van_pelt", "cable"] = "weighted",
    bind: bool = False,
    *,
    include_root: bool = True,
    property_name: str | None = None,
) -> ndarray | None:
    r"""Compute partition asymmetry for a NeuRosetta Tree.

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
        C1 and C2 are complete daughter-subtree cable lengths,
        including their connecting edges.

    For multifurcations, scores are averaged over all unordered
    pairs of daughter subtrees.

    Parameters
    ----------
    tree : _Tree
        NeuRosetta Tree object.
    mode : str, default "weighted"
        Partition asymmetry definition.
    bind : bool, default False
        If True, attach results as a vertex property and return None.
        Otherwise, return an ndarray.
    include_root : bool, default True
        Whether to include the root when it is a branching node.
    property_name : str or None, default None
        Vertex property name used when bind=True.
        Defaults to "partition_asymmetry_<mode>".

    Returns
    -------
    ndarray | None
        One asymmetry score per vertex, or None when bind=True.

    Notes
    -----
    Cable-based modes require the "Path_length" edge property
    containing true dendritic cable lengths.

    Euclidean distances must not be substituted for true cable
    lengths on reduced trees.
    """
    valid_modes = {"unweighted", "weighted", "van_pelt", "cable"}

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown partition asymmetry mode: {mode!r}. "
            f"Expected one of {sorted(valid_modes)}."
        )

    if mode in {"weighted", "cable"}:
        if not has_property(tree, "Path_length", "e"):
            raise ValueError(
                f"Mode {mode!r} requires the 'Path_length' "
                "edge property containing true dendritic cable lengths."
            )

    return partition_asymmetry(
        tree.graph,
        mode=mode,
        bind=bind,
        include_root=include_root,
        property_name=property_name,
    )


enrich_tree_graph_docstrings(globals())
