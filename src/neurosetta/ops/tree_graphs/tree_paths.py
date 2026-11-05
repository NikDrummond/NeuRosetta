"""Path retrieval on a NeuRosetta tree."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils.paths import path_edges as _path_edges
from ...utils.graph_utils.paths import path_exists as _path_exists
from ...utils.graph_utils.paths import path_vertices as _path_vertices
from .._doc_helpers import enrich_tree_graph_docstrings


def get_path(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return ordered node indices from ``source`` to ``target``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree. Edges point from parent to child.
    source, target : int
        Vertex indices in this tree. The same numbers on another tree are
        that tree's own vertices.
    directed : bool, optional
        Follow stored edges when True. When False, return the unique
        undirected path through the lowest common ancestor. The graph is
        not rerooted. By default True.

    Returns
    -------
    ndarray or None
        Indices including both endpoints. One element when
        ``source == target``. ``None`` when no directed path exists.

    Examples
    --------
    ``tree.get_path(10, 42)`` follows edges away from the root.
    ``tree.get_path(42, 10, directed=False)`` walks back through ancestors.
    """
    return _path_vertices(tree.graph, source, target, directed=directed)


def get_path_edges(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return traversal-ordered edge endpoints of shape ``(n_edges, 2)``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. A reversed walk lists endpoints in
        walk order, which can disagree with the stored parent-to-child
        edge. By default True.

    Returns
    -------
    ndarray or None
        Endpoint rows. Shape ``(0, 2)`` when ``source == target``.
        ``None`` when the path does not exist.
    """
    return _path_edges(tree.graph, source, target, directed=directed)


def path_exists(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> bool:
    """Return whether ``target`` is reachable from ``source``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Vertex indices. A vertex is reachable from itself.
    directed : bool, optional
        Stored direction when True. Every pair in a tree is reachable
        when False. By default True.

    Returns
    -------
    bool
        True when the path exists.
    """
    return _path_exists(tree.graph, source, target, directed=directed)


enrich_tree_graph_docstrings(globals())
