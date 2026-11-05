"""Morphological counts along one path.

Branch vertices are stored out-degree >= 2. A section is the cable between
consecutive landmarks. Landmarks are the root, every branch vertex, and
every leaf. That is the same partition counted by :func:`count_sections`
on a whole tree: transitive vertices sit inside a section and are not
landmarks.

A path that starts or ends on a transitive vertex still counts the section
it occupies. Moving from one landmark to the next counts as one section.
``source == target`` traverses nothing, so the count is 0.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import array, ndarray

from .retrieval import path_vertices


def _out_degree(g: Graph) -> ndarray:
    out_map = g.degree_property_map("out")
    return array(out_map.a, copy=True)


def path_branch_count(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
    include_endpoints: bool = False,
) -> int | None:
    """Count branch vertices on the path.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.
    include_endpoints : bool, optional
        Count ``source`` and ``target`` when they are branch vertices.
        By default False.

    Returns
    -------
    int or None
        Branch count, or ``None`` when the path does not exist.

    See Also
    --------
    branch_path
    path_section_count
    """
    vertices = path_vertices(g, source, target, directed=directed)
    if vertices is None:
        return None
    out_degree = _out_degree(g)
    last = int(vertices.size) - 1
    count = 0
    for position, vertex in enumerate(vertices):
        if out_degree[int(vertex)] < 2:
            continue
        if not include_endpoints and position in (0, last):
            continue
        count += 1
    return count


def path_section_count(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> int | None:
    """Count morphological sections travelled by the path.

    Landmarks are the root (in-degree 0), branch vertices (out-degree >= 2),
    and leaves (out-degree 0). Interior landmarks on the path, plus one,
    equal the number of landmark-to-landmark pieces the walk touches.
    A path that begins or ends between landmarks counts that partial piece
    as one section. A zero-length path counts 0.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    int or None
        Section count, or ``None`` when the path does not exist.

    See Also
    --------
    count_sections : section count for an entire tree
    path_branch_count
    """
    vertices = path_vertices(g, source, target, directed=directed)
    if vertices is None:
        return None
    if int(vertices.size) <= 1:
        return 0
    out_map = g.degree_property_map("out")
    in_map = g.degree_property_map("in")
    out_degree = array(out_map.a, copy=True)
    in_degree = array(in_map.a, copy=True)
    interior = 0
    for vertex in vertices[1:-1]:
        index = int(vertex)
        if out_degree[index] >= 2 or out_degree[index] == 0 or in_degree[index] == 0:
            interior += 1
    return interior + 1
