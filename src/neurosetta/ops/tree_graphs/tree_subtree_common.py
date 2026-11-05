"""Edge-length resolution shared by subtree and Sholl wrappers.

Not a Tree method. ``Path_length`` on a reduced tree is never replaced by
straight-line chords. On an unreduced tree each edge is one segment, so a
missing ``Path_length`` is the Euclidean length of that segment.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import empty, float64, ndarray

from ...core import _Tree

_LENGTHS = ("Path_length", "Euclidean_length")


def straight_line_edge_lengths(g: Graph) -> ndarray:
    """Euclidean length of every edge, indexed by graph-tool edge index."""
    x = g.vp["x"].a
    y = g.vp["y"].a
    z = g.vp["z"].a
    lengths = empty(g.num_edges(), dtype=float64)
    eindex = g.edge_index
    for edge in g.edges():
        src = int(edge.source())
        tgt = int(edge.target())
        dx = float(x[tgt]) - float(x[src])
        dy = float(y[tgt]) - float(y[src])
        dz = float(z[tgt]) - float(z[src])
        lengths[int(eindex[edge])] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return lengths


def resolve_tree_edge_weights(tree: _Tree, length_property: str) -> ndarray | None:
    """Return explicit weights, or None when *length_property* is already bound.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    length_property : {"Path_length", "Euclidean_length"}
        Property the caller asked for.

    Returns
    -------
    ndarray or None
        ``None`` tells the graph function to read the bound property.
        An array is edge-index aligned and is not written onto the graph.

    Raises
    ------
    ValueError
        If ``Path_length`` is missing on a reduced tree, or the name is unknown.
    """
    if length_property not in _LENGTHS:
        raise ValueError(
            f"Unknown length property {length_property!r}. Expected one of {_LENGTHS}."
        )
    g = tree.graph
    if length_property in g.ep:
        return None
    reduced = bool(tree.metadata.get("isReduced", False))
    if length_property == "Path_length" and reduced:
        raise ValueError(
            "Reduced tree is missing the 'Path_length' edge property. "
            "Physical cable length is not reconstructed from the straight-line "
            "chords of reduced edges."
        )
    return straight_line_edge_lengths(g)


def tree_length_units(tree: _Tree) -> str | None:
    """Return the tree's spatial unit string, or None when units are unset."""
    raw = tree.metadata.get("units")
    if raw is None:
        return None
    try:
        from ..units.tree_units import get_units

        return str(get_units(tree))
    except Exception:
        return str(raw)
