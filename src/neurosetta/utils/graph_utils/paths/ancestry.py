"""Parent links, ancestors, and lowest common ancestors.

Ancestry follows the stored parent-to-child orientation. It does not use
edge lengths. The deepest common ancestor is the one farthest from the
root in hop count, not in cable length.

Parent lookup uses the vertex in-neighbour accessor. It does not scan
every edge once per query. A multi-vertex LCA walks only the ancestor
chains of the requested vertices.
"""

from __future__ import annotations

from collections.abc import Iterable

from graph_tool.all import Graph
from numpy import array, flatnonzero, full, int64, ndarray, zeros

from ..tree_validation import as_vertex_index, validate_rooted_tree
from .retrieval import path_vertices


def _parent(g: Graph, vertex: int) -> int | None:
    parents = g.get_in_neighbors(vertex)
    if len(parents) == 0:
        return None
    return int(parents[0])


def _is_index(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    try:
        from numpy import integer
        from numpy import ndarray as np_ndarray

        if isinstance(value, integer):
            return True
        return isinstance(value, np_ndarray) and value.ndim == 0 and value.dtype.kind in "iu"
    except ImportError:
        return False


def _collect_vertices(vertices: int | Iterable[int] | None, others: tuple[int, ...]) -> list[int]:
    if vertices is None and not others:
        raise ValueError("lowest_common_ancestor requires at least one vertex.")
    if others:
        if vertices is None:
            sequence: Iterable[int] = others
        else:
            sequence = (vertices, *others)  # type: ignore[arg-type]
    elif _is_index(vertices):
        sequence = (vertices,)  # type: ignore[assignment]
    else:
        sequence = vertices  # type: ignore[assignment]
    collected: list[int] = []
    for item in sequence:
        if not _is_index(item):
            raise TypeError(
                "Vertex indices must be integers. Nested sequences are not expanded. "
                f"Got {type(item).__name__}."
            )
        collected.append(int(item))
    if not collected:
        raise ValueError("lowest_common_ancestor requires at least one vertex.")
    return collected


def parent_index(g: Graph, vertex: int) -> int | None:
    """Return the parent of ``vertex``.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    vertex : int
        Vertex index.

    Returns
    -------
    int or None
        Parent index. ``None`` for the root.

    See Also
    --------
    ancestor_indices
    is_ancestor
    """
    validate_rooted_tree(g)
    return _parent(g, as_vertex_index(g, vertex))


def ancestor_indices(
    g: Graph,
    vertex: int,
    *,
    include_self: bool = False,
) -> ndarray:
    """Return ancestors from the parent up to the root.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    vertex : int
        Starting vertex.
    include_self : bool, optional
        Put ``vertex`` first when True. By default False.

    Returns
    -------
    ndarray
        int64 indices. Empty for the root when ``include_self`` is False.
        Order is immediate parent, then its parent, ending at the root.

    See Also
    --------
    parent_index
    lowest_common_ancestor
    """
    validate_rooted_tree(g)
    current = as_vertex_index(g, vertex)
    chain: list[int] = []
    if include_self:
        chain.append(current)
    parent = _parent(g, current)
    while parent is not None:
        chain.append(parent)
        parent = _parent(g, parent)
    return array(chain, dtype=int64)


def is_ancestor(g: Graph, ancestor: int, descendant: int) -> bool:
    """Return whether ``ancestor`` lies on the path from ``descendant`` to the root.

    The test is inclusive: ``is_ancestor(g, v, v)`` is True. A directed
    path from ``ancestor`` to ``descendant`` exists exactly when this is
    True, because edges point away from the root.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    ancestor, descendant : int
        Vertex indices.

    Returns
    -------
    bool
        True when ``ancestor`` is ``descendant`` or a strict ancestor.

    See Also
    --------
    path_exists
    lowest_common_ancestor
    """
    validate_rooted_tree(g)
    anc = as_vertex_index(g, ancestor)
    current = as_vertex_index(g, descendant)
    while True:
        if current == anc:
            return True
        parent = _parent(g, current)
        if parent is None:
            return False
        current = parent


def lowest_common_ancestor(
    g: Graph,
    vertices: int | Iterable[int] | None = None,
    *others: int,
) -> int:
    """Return the deepest vertex ancestral to every supplied vertex.

    Depth is the number of edges from the root, not cable length. Any
    nonempty set of vertices in a connected rooted tree has one LCA.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. The root index need not be 0.
    vertices : int or iterable of int, optional
        One index, or a list, tuple, ndarray, or other finite iterable.
        Combined with ``others`` when those are passed.
    *others : int
        Extra indices. ``lowest_common_ancestor(g, 3, 4, 5)`` and
        ``lowest_common_ancestor(g, [3, 4, 5])`` are the same query.

    Returns
    -------
    int
        Deepest common ancestor. A single vertex is its own LCA.
        Duplicate indices do not change the result.

    Raises
    ------
    ValueError
        If no vertices are supplied, or an index is out of range.
    TypeError
        If an entry is not an integer index.

    Examples
    --------
    For the tree ``0-1-4`` with children 6 and 7 of vertex 4:

    ``lowest_common_ancestor(g, 6, 7) == 4``

    ``lowest_common_ancestor(g, 3, 6, 7) == 1`` when 3 is the other child of 1.

    See Also
    --------
    is_ancestor
    ancestor_indices
    """
    requested = _collect_vertices(vertices, others)
    validate_rooted_tree(g)
    unique: list[int] = []
    seen: set[int] = set()
    for raw in requested:
        index = as_vertex_index(g, raw)
        if index not in seen:
            seen.add(index)
            unique.append(index)
    if len(unique) == 1:
        return unique[0]

    chains = [_chain_from_root(g, vertex) for vertex in unique]
    limit = min(len(chain) for chain in chains)
    shared = chains[0][0]
    for position in range(limit):
        head = chains[0][position]
        if all(chain[position] == head for chain in chains):
            shared = head
        else:
            break
    return int(shared)


def _chain_from_root(g: Graph, vertex: int) -> list[int]:
    chain = [vertex]
    parent = _parent(g, vertex)
    while parent is not None:
        chain.append(parent)
        parent = _parent(g, parent)
    chain.reverse()
    return chain


def parent_depth_tables(g: Graph) -> tuple[ndarray, ndarray, int]:
    """Return parent indices, hop depth, and the root.

    ``parents[root]`` is -1. Built in one edge scan plus a walk from the
    root. Callers that already validated the tree reuse this for a batch
    of ancestor queries. It is not cached on the graph.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.

    Returns
    -------
    parents : ndarray
        int64 parent index, or -1 at the root.
    depth : ndarray
        int64 hop distance from the root.
    root : int
        The unique in-degree 0 vertex.
    """
    n = int(g.num_vertices())
    parents = full(n, -1, dtype=int64)
    for edge in g.edges():
        parents[int(edge.target())] = int(edge.source())
    roots = flatnonzero(parents < 0)
    root = int(roots[0])
    depth = zeros(n, dtype=int64)
    children: list[list[int]] = [[] for _ in range(n)]
    for vertex, parent in enumerate(parents):
        if parent >= 0:
            children[int(parent)].append(vertex)
    stack = [root]
    while stack:
        vertex = stack.pop()
        for child in children[vertex]:
            depth[child] = depth[vertex] + 1
            stack.append(child)
    return parents, depth, root


def lca_from_tables(parents: ndarray, depth: ndarray, left: int, right: int) -> int:
    """Return the LCA of two vertices using parent and depth tables."""
    a = int(left)
    b = int(right)
    while depth[a] > depth[b]:
        a = int(parents[a])
    while depth[b] > depth[a]:
        b = int(parents[b])
    while a != b:
        a = int(parents[a])
        b = int(parents[b])
    return a


def branch_path(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
    include_endpoints: bool = False,
) -> ndarray | None:
    """Return branch vertices along the path, in traversal order.

    A branch vertex has out-degree at least 2 in the stored orientation.
    That definition does not depend on which way the path is walked.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.
    include_endpoints : bool, optional
        Keep ``source`` and ``target`` when they are branch vertices.
        By default False.

    Returns
    -------
    ndarray or None
        int64 branch indices. Empty when the path exists and contains no
        selected branch vertices. ``None`` when the path does not exist.

    See Also
    --------
    path_vertices
    path_branch_count
    """
    vertices = path_vertices(g, source, target, directed=directed)
    if vertices is None:
        return None
    out_map = g.degree_property_map("out")
    out_degree = array(out_map.a, copy=True)
    chosen: list[int] = []
    last = int(vertices.size) - 1
    for position, vertex in enumerate(vertices):
        index = int(vertex)
        if out_degree[index] < 2:
            continue
        if not include_endpoints and position in (0, last):
            continue
        chosen.append(index)
    return array(chosen, dtype=int64)
