"""Functions for editing trees."""

from __future__ import annotations

from graph_tool.all import Graph

from ...core import _Tree
from ...utils.graph_utils import reduce_graph, reroot_graph
from .._doc_helpers import enrich_tree_graph_docstrings


def reduce_tree(tree: _Tree, inplace: bool = False) -> Graph | None:
    """Remove transitive nodes from the tree graph.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    inplace : bool, optional
        If True, replace the tree's graph with the reduced graph in place and
        update metadata. If False, return the reduced graph. By default False.

    Returns
    -------
    Graph | None
        Reduced graph if inplace=False, otherwise None.

    Notes
    -----
    When *inplace* is True and synapses are already mapped, edge locations are
    transferred onto reduced section edges using cable-distance provenance
    (raw / mapped coordinates and ``distance_to_tree`` are unchanged). Unmapped
    synapses are preserved as raw observations.

    Recommended workflow for exact morphology-relative positions::

        tree.set_synapses(data)
        tree.map_synapses()
        tree.get_reduced_tree(inplace=True)
    """
    from .tree_path_lengths import get_edge_length
    from .tree_synapses import _bind_synapses_gp, get_synapses

    syn = get_synapses(tree)
    # Ensure Path_length exists (required by reduce_graph).
    get_edge_length(tree, bind=True)

    if inplace:
        g, reduction_map = reduce_graph(tree.graph, return_mapping=True)
        tree.graph = g
        if syn is not None:
            syn = syn.copy()
            if syn.is_mapped:
                syn.remap_edges_through_reduction(reduction_map)
            _bind_synapses_gp(tree, syn)
        return None

    g = reduce_graph(tree.graph.copy())
    return g


def reroot_tree(tree: _Tree, root: int, inplace: bool = False) -> Graph | None:
    """Reroot the tree to a new root vertex.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int
        Vertex index to use as the new root.
    inplace : bool, optional
        If True, replace the tree's graph with the rerooted graph in place.
        If False, return the rerooted graph. By default False.

    Returns
    -------
    Graph | None
        Rerooted graph if inplace=False, otherwise None.

    Notes
    -----
    Raw synapse coordinates are unchanged. Edge mappings are invalidated when
    *inplace* is True because edge indices are rebuilt by rerooting.
    """
    from .tree_synapses import _bind_synapses_gp, get_synapses, invalidate_synapse_mapping

    syn = get_synapses(tree)
    g = reroot_graph(tree.graph, root)

    if inplace:
        tree.graph = g
        if syn is not None:
            _bind_synapses_gp(tree, syn)
            invalidate_synapse_mapping(tree)
        return None
    return g


enrich_tree_graph_docstrings(globals())
