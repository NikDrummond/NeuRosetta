"""Property values sampled along a path."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils.paths import path_edge_values, path_vertex_values
from .._doc_helpers import enrich_tree_graph_docstrings


def get_path_vertex_values(
    tree: _Tree,
    source: int,
    target: int,
    vp: str,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return a vertex property along the path.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    vp : str
        Vertex property name. Numeric and string properties keep their type.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Values in walk order. ``None`` when the path does not exist.
    """
    return path_vertex_values(tree.graph, source, target, vp, directed=directed)


def get_path_edge_values(
    tree: _Tree,
    source: int,
    target: int,
    ep: str,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return an edge property along the path.

    Reversed walks read the stored edge. They do not look up the reversed
    endpoint pair as a directed edge.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    ep : str
        Edge property name.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Values in walk order. Empty when ``source == target``. ``None``
        when the path does not exist.
    """
    return path_edge_values(tree.graph, source, target, ep, directed=directed)


enrich_tree_graph_docstrings(globals())
