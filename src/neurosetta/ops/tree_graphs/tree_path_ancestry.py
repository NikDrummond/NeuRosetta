"""Ancestry queries on a NeuRosetta tree."""

from __future__ import annotations

from collections.abc import Iterable

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils.paths import ancestor_indices as _ancestor_indices
from ...utils.graph_utils.paths import branch_path as _branch_path
from ...utils.graph_utils.paths import is_ancestor as _is_ancestor
from ...utils.graph_utils.paths import lowest_common_ancestor as _lowest_common_ancestor
from ...utils.graph_utils.paths import parent_index as _parent_index
from .._doc_helpers import enrich_tree_graph_docstrings


def get_parent_index(tree: _Tree, vertex: int) -> int | None:
    """Return the parent of ``vertex``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    vertex : int
        Vertex index.

    Returns
    -------
    int or None
        Parent index. ``None`` at the root.
    """
    return _parent_index(tree.graph, vertex)


def get_ancestor_indices(
    tree: _Tree,
    vertex: int,
    *,
    include_self: bool = False,
) -> ndarray:
    """Return ancestors from the parent up to the root.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    vertex : int
        Starting vertex.
    include_self : bool, optional
        Include ``vertex`` first when True. By default False.

    Returns
    -------
    ndarray
        int64 indices ending at the root.
    """
    return _ancestor_indices(tree.graph, vertex, include_self=include_self)


def is_ancestor(tree: _Tree, ancestor: int, descendant: int) -> bool:
    """Return whether ``ancestor`` is on the path from ``descendant`` to the root.

    Inclusive: a vertex is an ancestor of itself. Independent of edge length.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    ancestor, descendant : int
        Vertex indices.

    Returns
    -------
    bool
        True when ``ancestor`` is ``descendant`` or a strict ancestor.
    """
    return _is_ancestor(tree.graph, ancestor, descendant)


def get_lowest_common_ancestor(
    tree: _Tree,
    vertices: int | Iterable[int] | None = None,
    *others: int,
) -> int:
    """Return the deepest common ancestor of any nonempty vertex set.

    Depth is hop distance from the root, not cable length.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    vertices : int or iterable of int, optional
        One index, or a list, tuple, or array of indices.
    *others : int
        Further indices. ``tree.get_lowest_common_ancestor(10, 20, 30)``
        and ``tree.get_lowest_common_ancestor([10, 20, 30])`` match.

    Returns
    -------
    int
        Lowest common ancestor. One vertex returns itself.

    Raises
    ------
    ValueError
        If no vertices are given.
    """
    return _lowest_common_ancestor(tree.graph, vertices, *others)


def get_branch_path(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
    include_endpoints: bool = False,
) -> ndarray | None:
    """Return branch nodes along the path, in walk order.

    A branch node has at least two children. Endpoints are omitted unless
    ``include_endpoints`` is True.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.
    include_endpoints : bool, optional
        Keep endpoints that are branch nodes. By default False.

    Returns
    -------
    ndarray or None
        Branch indices. Empty when the path has none. ``None`` when the
        path does not exist.
    """
    return _branch_path(
        tree.graph,
        source,
        target,
        directed=directed,
        include_endpoints=include_endpoints,
    )


enrich_tree_graph_docstrings(globals())
