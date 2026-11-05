"""Coordinates, endpoint displacement, and tortuosity along a path.

Endpoint distance is the straight line between the two vertex coordinates.
It is not the sum of ``Euclidean_length`` and it does not depend on edge
direction. Tortuosity divides physical cable by that displacement.
``Path_length`` is required unless another property is passed explicitly.
On a reduced tree, ``Path_length`` is the stored section cable. The
straight chord between the endpoints is still the denominator, so the
ratio describes the stored cable against the current endpoints. It is not
a claim that the reduced edge geometry is the original curve.
"""

from __future__ import annotations

from graph_tool import EdgePropertyMap
from graph_tool.all import Graph
from numpy import linalg, nan, ndarray

from ..coordinates import vertex_coordinates
from ..tree_validation import as_vertex_index, validate_rooted_tree
from .distances import path_length
from .retrieval import path_vertices


def path_coordinates(
    g: Graph,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> ndarray | None:
    """Return ``(n_vertices, 3)`` coordinates in path order.

    Uses vertex properties ``x``, ``y``, and ``z``. No points are inserted
    between vertices.

    Parameters
    ----------
    g : Graph
        Directed rooted tree with coordinates.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        Coordinates of the path vertices. ``None`` when the path does not
        exist.

    See Also
    --------
    path_vertices
    path_euclidean_distance
    """
    vertices = path_vertices(g, source, target, directed=directed)
    if vertices is None:
        return None
    return vertex_coordinates(g, subset=vertices.tolist(), SoA=False)


def path_euclidean_distance(g: Graph, source: int, target: int) -> float:
    """Return the straight-line distance between two vertex coordinates.

    Direction is irrelevant: the value is the Euclidean norm of the
    coordinate difference. It is not the sum of ``Euclidean_length``.

    Parameters
    ----------
    g : Graph
        Directed rooted tree with ``x``, ``y``, and ``z``.
    source, target : int
        Endpoint indices.

    Returns
    -------
    float
        Displacement. 0 when both indices are the same vertex.

    See Also
    --------
    path_length
    path_tortuosity
    """
    validate_rooted_tree(g)
    src = as_vertex_index(g, source)
    tgt = as_vertex_index(g, target)
    coords = vertex_coordinates(g, subset=[src, tgt], SoA=False)
    return float(linalg.norm(coords[1] - coords[0]))


def path_tortuosity(
    g: Graph,
    source: int,
    target: int,
    ep: str | EdgePropertyMap = "Path_length",
    *,
    directed: bool = True,
) -> float | None:
    """Return cable length divided by endpoint displacement.

    .. math::

        \\tau(u, v) = L(u, v) / \\lVert x(v) - x(u) \\rVert

    ``L`` is the sum of ``ep`` along the unique path. The default ``ep``
    is ``Path_length``. ``Euclidean_length`` is not substituted.

    Parameters
    ----------
    g : Graph
        Directed rooted tree with coordinates and the requested length
        property.
    source, target : int
        Path endpoints.
    ep : str or EdgePropertyMap, optional
        Physical length property. By default ``"Path_length"``.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    float or None
        Tortuosity. ``None`` when the directed path does not exist.
        ``nan`` when ``source == target``, and when the endpoints occupy
        the same point, because the ratio is undefined.

    Notes
    -----
    A reduced edge stores the original section's ``Path_length`` and a
    straight endpoint chord. Tortuosity then compares that stored cable
    with the chord between the path's endpoints. Summing ``Euclidean_length``
    instead measures the reduced polyline, not the original cable.

    See Also
    --------
    path_length
    path_euclidean_distance
    """
    cable = path_length(g, source, target, ep, directed=directed)
    if cable is None:
        return None
    if as_vertex_index(g, source) == as_vertex_index(g, target):
        return float(nan)
    displacement = path_euclidean_distance(g, source, target)
    if displacement == 0.0:
        return float(nan)
    return float(cable) / displacement
