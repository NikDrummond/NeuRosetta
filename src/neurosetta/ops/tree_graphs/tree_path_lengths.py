"""Functions for distance calculations on trees."""

from typing import Literal

from numpy import array, ndarray

from ...core import _Tree
from ...utils.geometry_utils import euclidean_distance
from ...utils.graph_utils import (
    bind_edge_property,
    edge_coordinates,
    g_has_property,
    pairwise_path_lengths,
    path_cumulative_length,
    path_edge_count,
    path_length,
    path_lengths_to_vertices,
    root_distance,
    source_distances,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import check_reduced

LengthType = Literal["Path", "Euclidean"]


def get_edge_length(tree: _Tree, bind: bool = True, recalculate: bool = False) -> ndarray | None:
    """Get length of edges.

    Row ``i`` is graph-tool edge index ``i`` (the index into ``ep.*.a``),
    not necessarily the row order of ``Graph.get_edges()``.

    When ``Path_length`` is present it is returned, including on a reduced
    tree, where that property is the accumulated cable of the collapsed
    section. That cable is not the Euclidean chord between the surviving
    vertices. ``Euclidean_length`` is used only when ``Path_length`` is
    absent. If neither property exists, lengths are the Euclidean distances
    between the current endpoints and are stored as ``Path_length`` on a
    non-reduced tree or ``Euclidean_length`` on a reduced tree.
    ``recalculate=True`` always recomputes those endpoint distances and
    ignores stored cable.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    bind : bool, optional
        If True, bind the lengths as an edge property and return None.
        If False, return the lengths array. By default True.
    recalculate : bool, optional
        If True, recompute edge lengths from coordinates instead of reading
        existing edge properties. By default False.

    Returns
    -------
    ndarray | None
        If bind=False, returns array of edge lengths. Otherwise returns None.
    """

    lengths = None
    if not recalculate:
        if g_has_property(tree.graph, "Path_length", "e"):
            lengths = tree.graph.ep["Path_length"].a
        elif g_has_property(tree.graph, "Euclidean_length", "e"):
            lengths = tree.graph.ep["Euclidean_length"].a

    if lengths is None:
        (x1, y1, z1), (x2, y2, z2) = edge_coordinates(tree.graph, SoA=True)
        lengths = euclidean_distance(x1, y1, z1, x2, y2, z2)

        if bind:
            # base property name on if this is a reduced graph
            p_name = "Euclidean_length" if check_reduced(tree) else "Path_length"
            bind_edge_property(
                tree.graph,
                property_name=p_name,
                property_dtype="double",
                property_data=lengths,
            )
    return lengths


def get_total_cable_length(tree: _Tree, length_type: LengthType = "Path") -> float:
    """Compute total cable length of the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    length_type : {"Path", "Euclidean"}, optional
        Edge length property to sum. ``"Path"`` uses ``Path_length``;
        ``"Euclidean"`` uses ``Euclidean_length``. By default ``"Path"``.

    Returns
    -------
    float
        Total cable length (sum of all edge lengths).
    """
    prop_name = "Path_length" if length_type == "Path" else "Euclidean_length"

    if g_has_property(tree.graph, prop_name, "e"):
        lengths = tree.graph.ep[prop_name].a
    else:
        lengths = get_edge_length(tree, bind=False, recalculate=True)

    return float(lengths.sum())


def get_root_distance(
    tree: _Tree,
    length_type: LengthType = "Path",
    bind: bool = True,
) -> ndarray | None:
    """Compute path distance from the root to every node.

    Distances are shortest-path sums of edge lengths from the tree root
    (in-degree == 0). ``length_type="Path"`` is stored as ``Root_distance``.
    ``length_type="Euclidean"`` is stored as ``Root_euclidean_distance``.
    The two names are distinct so a cached path distance is not read back
    as a Euclidean distance.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    length_type : {"Path", "Euclidean"}, optional
        Edge length property used as path weights. ``"Path"`` uses
        ``Path_length``; ``"Euclidean"`` uses ``Euclidean_length``.
        If the requested property is missing, edge lengths are computed
        from coordinates and bound under that name. By default ``"Path"``.
    bind : bool, optional
        If True, bind distances under the metric-specific vertex property
        and return None. If False, return the distance array and do not
        write a vertex property. By default True.

    Returns
    -------
    ndarray | None
        Per-node distance from the root if ``bind=False``, otherwise None.
        The root node has distance ``0``.
    """
    prop_name = "Path_length" if length_type == "Path" else "Euclidean_length"

    if not g_has_property(tree.graph, prop_name, "e"):
        lengths = get_edge_length(tree, bind=False, recalculate=True)
        bind_edge_property(
            tree.graph,
            property_name=prop_name,
            property_dtype="double",
            property_data=lengths,
        )

    rd = root_distance(tree.graph, prop_name)

    if bind:
        # Historical name stays ``Root_distance`` for physical cable.
        # Euclidean results use a separate map so the two cannot collide.
        bound_name = "Root_distance" if length_type == "Path" else "Root_euclidean_distance"
        tree.graph.vp[bound_name] = rd
        return None
    # ``rd`` is not stored on the graph. ``rd.a`` is a view into that map and
    # becomes invalid as soon as ``rd`` is freed, so return an owned copy.
    return array(rd.a, dtype=float, copy=True)


def _source_distance_name(source: int, ep, directed: bool) -> str:
    if isinstance(ep, str):
        weight = ep
    elif ep is None:
        weight = "auto"
    else:
        weight = "custom"
    direction = "directed" if directed else "undirected"
    return f"path_distance_from_{int(source)}_{weight}_{direction}"


def get_path_length(tree: _Tree, source: int, target: int, ep=None, *, directed: bool = True):
    """Return the weighted length of the path from ``source`` to ``target``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Vertex indices in this tree. 0 when they are equal.
    ep : str or EdgePropertyMap or None, optional
        ``None`` uses ``Path_length``, then ``Euclidean_length``. An
        explicit name is never replaced. Coordinates are not used to
        fill a missing property. By default None.
    directed : bool, optional
        Follow parent-to-child edges when True. By default True.

    Returns
    -------
    float or None
        Path weight, or ``None`` when that directed path does not exist.
    """
    return path_length(tree.graph, source, target, ep, directed=directed)


def get_path_edge_count(tree: _Tree, source: int, target: int, *, directed: bool = True):
    """Return the number of edges on the path.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Vertex indices. The count is 0 when they are equal.
    directed : bool, optional
        Follow stored edge direction when True. By default True.

    Returns
    -------
    int or None
        Edge count, or ``None`` when the path does not exist.
    """
    return path_edge_count(tree.graph, source, target, directed=directed)


def get_path_cumulative_length(
    tree: _Tree,
    source: int,
    target: int,
    ep=None,
    *,
    directed: bool = True,
):
    """Return cumulative path weight at each vertex from ``source``.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    ep : str or EdgePropertyMap or None, optional
        Weight property. Same rules as :func:`get_path_length`.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    ndarray or None
        float64 running sum, starting at 0. ``None`` when the path does
        not exist. The last entry equals :func:`get_path_length`.
    """
    return path_cumulative_length(tree.graph, source, target, ep, directed=directed)


def get_source_distances(
    tree: _Tree,
    source: int,
    ep=None,
    *,
    directed: bool = True,
    bind: bool = False,
    property_name: str | None = None,
):
    """Return weighted distances from one node to every node.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source : int
        Source vertex. This is not required to be the root.
    ep : str or EdgePropertyMap or None, optional
        Weight property. ``None`` uses ``Path_length``, then
        ``Euclidean_length``.
    directed : bool, optional
        Stored direction when True. Unreachable nodes are ``inf``.
        By default True.
    bind : bool, optional
        Store the vertex property and return None when True. The name is
        ``property_name`` or ``path_distance_from_{source}_{weight}_{direction}``.
        ``Root_distance`` is not reused. By default False.
    property_name : str or None, optional
        Vertex property written when ``bind`` is True.

    Returns
    -------
    ndarray or None
        float64 distances when ``bind`` is False. The source is 0.
    """
    dist = source_distances(tree.graph, source, ep, directed=directed)
    if bind:
        name = property_name or _source_distance_name(source, ep, directed)
        tree.graph.vp[name] = dist
        return None
    from numpy import array

    return array(dist.a, dtype=float, copy=True)


def get_path_lengths_to_vertices(
    tree: _Tree,
    source: int,
    targets,
    ep=None,
    *,
    directed: bool = True,
):
    """Return distances from ``source`` to each target, in target order.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source : int
        Source vertex.
    targets : sequence of int
        Target indices in this tree. Indices are not shared with other trees.
    ep : str or EdgePropertyMap or None, optional
        Weight property.
    directed : bool, optional
        Stored direction when True. Unreachable targets are ``inf``.

    Returns
    -------
    ndarray
        float64 distances.
    """
    return path_lengths_to_vertices(tree.graph, source, targets, ep, directed=directed)


def get_pairwise_path_lengths(tree: _Tree, pairs, ep=None, *, directed: bool = True):
    """Return one distance per ``(source, target)`` pair.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    pairs : array-like, shape (n_pairs, 2)
        Vertex indices in this tree.
    ep : str or EdgePropertyMap or None, optional
        Weight property.
    directed : bool, optional
        Stored direction when True. Unreachable pairs are ``inf``.

    Returns
    -------
    ndarray
        float64 distances in pair order.
    """
    return pairwise_path_lengths(tree.graph, pairs, ep, directed=directed)


enrich_tree_graph_docstrings(globals())
