"""Functions to modify structure of graphs"""

from __future__ import annotations

from graph_tool.all import Graph, GraphView
from numpy import concatenate, empty, vstack

from ...core.tree_helpers import copy_tree_identity
from .coordinates import vertex_coordinates
from .gt_properties import g_has_property, raise_internal_property_missing
from .node_types import infer_node_types
from .reduction_map import ReductionMap
from .traversals import ReduceVisitor, bf_iterator, dfsearch
from .vertex_inds import branch_indices, core_indices, root_index


def _cable_length_prop_name(g: Graph) -> str:
    """Return the edge property used as physical cable length."""
    if g_has_property(g, "Path_length", "e"):
        return "Path_length"
    if g_has_property(g, "Euclidean_length", "e"):
        return "Euclidean_length"
    raise_internal_property_missing(g, "Path_length", "e")
    return "Path_length"  # unreachable


def reduce_graph(
    g: Graph,
    *,
    return_mapping: bool = False,
) -> Graph | tuple[Graph, ReductionMap]:
    """Reduce a graph by collapsing transitive vertices.

    Parameters
    ----------
    g : Graph
        Input graph with vertex properties 'x', 'y', 'z', 'radius', 'node_type',
        edge property ``Path_length`` or ``Euclidean_length``, and graph
        properties 'ID' and 'metadata'.
    return_mapping : bool, optional
        If True, also return a :class:`~neurosetta.utils.graph_utils.reduction_map.ReductionMap`
        describing which original edges compose each reduced section.
        By default False (unchanged return type for existing callers).

    Returns
    -------
    Graph or (Graph, ReductionMap)
        Reduced graph with only branch and leaf vertices (plus root), preserving
        path lengths, coordinates, radii, node types, and graph metadata.
        When *return_mapping* is True, also returns the section provenance map.
    """
    # we need path a bunch of properties
    raise_internal_property_missing(g, "metadata", "g")
    raise_internal_property_missing(g, "ID", "g")
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")
    raise_internal_property_missing(g, "radius", "v")
    raise_internal_property_missing(g, "node_type", "v")
    length_prop = _cable_length_prop_name(g)

    # get root
    root = root_index(g)

    ### stops are all leaves and branches excluding the root
    stops = core_indices(g, include_root=False)
    # starts are all branches including the root
    starts = branch_indices(g)
    # make sure we have the root
    if root not in starts:
        starts = concatenate([starts, [root]])

    # Traversal
    # generate visitor
    vis = dfsearch(
        g,
        ReduceVisitor,
        {"graph": g, "starts": starts, "stops": stops, "length_prop": length_prop},
        root=root,
        bind=False,
    )

    # create edge list
    edges = vstack((vis.edge_source, vis.edge_target)).T
    # make graph from edge list
    g_red = Graph(edges, hashed=True, hash_type=int)
    # add path lengths to edges (reduced sections always use Path_length)
    g_red.ep["Path_length"] = g_red.new_ep("float", vis.path_lengths)
    # add coordinates to verts
    coords = vertex_coordinates(g)[g_red.vp["ids"].a]
    g_red.vp["x"] = g_red.new_vp("double", coords[:, 0])
    g_red.vp["y"] = g_red.new_vp("double", coords[:, 1])
    g_red.vp["z"] = g_red.new_vp("double", coords[:, 2])
    # add identity (from original) — preserves int/str ID and artifact name
    copy_tree_identity(g, g_red)
    # add metadata and update reduced
    meta = g.gp["metadata"].copy()
    meta["file_path"] = ""
    meta["isReduced"] = True
    g_red.gp["metadata"] = g_red.new_gp("object", meta)
    # radius
    g_red.vp["radius"] = g_red.new_vp("double", g.vp["radius"].a[g_red.vp["ids"].a])
    # node type
    g_red.vp["node_type"] = g_red.new_vp("int", g.vp["node_type"].a[g_red.vp["ids"].a])

    if not return_mapping:
        return g_red

    reduction_map = ReductionMap.from_sections(
        vis.section_edge_indices,
        vis.section_edge_lengths,
        n_original_edges=g.num_edges(),
        n_original_vertices=g.num_vertices(),
    )
    return g_red, reduction_map


def _undirected_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _migrate_edge_length(g_src: Graph, g_dst: Graph, prop_name: str) -> None:
    """Copy an edge length across rerooting by undirected endpoint identity.

    Reversed edges keep the stored length. Lengths are not recomputed from
    coordinates, so physical ``Path_length`` is not replaced by a chord.
    """
    if prop_name not in g_src.ep or g_dst.num_edges() == 0:
        return
    lookup: dict[tuple[int, int], float] = {}
    for row in g_src.get_edges([g_src.ep[prop_name]]):
        lookup[_undirected_key(int(row[0]), int(row[1]))] = float(row[2])
    ids = g_dst.vp["ids"].a
    new_edges = g_dst.get_edges()
    lengths = empty(new_edges.shape[0], dtype=float)
    for i, (src, tgt) in enumerate(new_edges):
        key = _undirected_key(int(ids[int(src)]), int(ids[int(tgt)]))
        try:
            lengths[i] = lookup[key]
        except KeyError as exc:
            raise ValueError(
                f"Cannot migrate edge property {prop_name!r} while rerooting: "
                f"no original edge between vertices {key[0]} and {key[1]}."
            ) from exc
    g_dst.ep[prop_name] = g_dst.new_ep("double", lengths)


def reroot_graph(g: Graph, root: int) -> Graph:
    """Reroot a graph at a specified vertex.

    Parameters
    ----------
    g : Graph
        Input graph with vertex properties 'x', 'y', 'z', 'radius', 'node_type',
        and graph properties 'ID' and 'metadata'. Edge properties
        ``Path_length`` and ``Euclidean_length`` are copied when present.
    root : int
        Vertex index to use as new root.

    Returns
    -------
    Graph
        New graph rooted at the specified vertex with all properties migrated.
        Total ``Path_length`` is unchanged: reversed edges keep their stored
        cable length.
    """
    # we need path a bunch of properties
    raise_internal_property_missing(g, "metadata", "g")
    raise_internal_property_missing(g, "ID", "g")
    raise_internal_property_missing(g, "x", "v")
    raise_internal_property_missing(g, "y", "v")
    raise_internal_property_missing(g, "z", "v")
    raise_internal_property_missing(g, "radius", "v")
    raise_internal_property_missing(g, "node_type", "v")

    # Undirected GraphView to get edge list from new root
    g_view = GraphView(g, directed=False)
    # edge list from new root (BFS)
    edges = bf_iterator(g_view, root, array=True)
    # new graph
    g_new = Graph(edges, hashed=True, hash_type="int")

    ### migrate properties

    # identity (ID + name)
    copy_tree_identity(g_view, g_new)
    # metadata
    g_new.gp["metadata"] = g_new.new_gp("object", g_view.gp["metadata"])
    # coordinates
    ids = g_new.vp["ids"].a
    g_new.vp["x"] = g_new.new_vp("double", g_view.vp["x"].a[ids])
    g_new.vp["y"] = g_new.new_vp("double", g_view.vp["y"].a[ids])
    g_new.vp["z"] = g_new.new_vp("double", g_view.vp["z"].a[ids])
    # node_type (re-infer from rerooted directed topology)
    g_new.vp["node_type"] = g_new.new_vp("int", infer_node_types(g_new))
    # radius
    g_new.vp["radius"] = g_new.new_vp("double", g_view.vp["radius"].a[g_new.vp["ids"].a])
    # Cable lives on the undirected segment. Do not recompute it from the
    # straight-line chord after edges reverse.
    _migrate_edge_length(g, g_new, "Path_length")
    _migrate_edge_length(g, g_new, "Euclidean_length")

    return g_new
