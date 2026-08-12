"""Module for breadth and depth first traversals on graphs."""
from __future__ import annotations

from typing import Iterable
from numpy import ndarray, asarray, float64

from graph_tool.all import (
    BFSVisitor,
    bfs_search,
    DFSVisitor,
    dfs_search,
    Graph,
    bfs_iterator,
    dfs_iterator,
)

from ..geometry_utils.angles import angle, angular_mean, angular_var


### BFS traversals


def bfsearch(
    g: Graph,
    visitor: BFSVisitor,
    init_kwargs: dict | None = None,
    init_vertex_properties: dict | None = None,
    init_edge_properties: dict | None = None,
    root: int = 0,
    bind: bool = True,
):
    """Wrapper for graph_tool's bfs_search that initializes properties and binds
    them to the graph if bind = True.

    Parameters
    ----------
    g : graph_tool.Graph
        Graph to search.
    visitor : graph_tool.BFSVisitor
        Visitor object that defines the behaviour of the search.
    init_kwargs : dict, optional
        Keyword arguments passed to the visitor constructor.
    init_vertex_properties : dict, optional
        Dictionary of vertex property name and type pairs to initialize before
        the search. For example, {"depth": "int"} will create a vertex property
        called "depth" of type int and pass it to the visitor.
    init_edge_properties : dict, optional
        Dictionary of edge property name and type pairs to initialize before
        the search.
    root : int, optional
        Index of the root vertex to start the search from, by default 0.
    bind : bool, optional
        Whether to bind the initialized properties to the graph after the search,
        by default True. If False, the properties are not bound to the graph and
        instead returned as a dictionary.

    Returns
    -------
    visitor : BFSVisitor
        The visitor instance after the search.
    dict (if bind = False and properties were initialized)
        A dictionary of property name and property pairs for all initialized
        properties. For example, {"depth": vdepth} where vdepth is a vertex
        property map containing the depth of each vertex from the root.
    """
    if init_kwargs is None:
        init_kwargs = {}
    if init_vertex_properties is None:
        init_vertex_properties = {}
    if init_edge_properties is None:
        init_edge_properties = {}

    # initialized properties
    vprops = {name: g.new_vp(ptype) for name, ptype in init_vertex_properties.items()}
    eprops = {name: g.new_ep(ptype) for name, ptype in init_edge_properties.items()}
    properties = {**vprops, **eprops}

    # make visitor
    vis = visitor(**init_kwargs, **properties)
    # bfs search
    bfs_search(g, root, vis)

    # bind properties to graph or return them as a dictionary
    if bool(properties):
        if bind:
            for name, prop in vprops.items():
                g.vertex_properties[name] = prop
            for name, prop in eprops.items():
                g.edge_properties[name] = prop
        else:
            return properties
    return vis


def bf_iterator(g: Graph, root: int = 0, array: bool = True) -> Iterable | ndarray:
    """Return breadth first search iterator or array of visited edges from root.

    Parameters
    ----------
    g : Graph
        Graph object.
    root : int, optional
        Root node index, by default 0.
    array : bool, optional
        If False, returns iterator over edges, otherwise returns a numpy.ndarray,
        by default True.

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in breadth first search traversal order.
    """
    return bfs_iterator(g, root, array)


### BFS visitors


class TreeDepthVisitor(BFSVisitor):
    """Visitor class to obtain node depths in tree."""

    def __init__(self, depth):
        """Initialize visitor with depth property map.

        Parameters
        ----------
        depth : graph_tool.VertexPropertyMap
            Vertex property map to store depth values.
        """
        self.depth = depth

    def tree_edge(self, e):
        self.depth[e.target()] = self.depth[e.source()] + 1


class SubtreeMaskVisitor(BFSVisitor):
    """Visitor class to mark edges and vertices in a subtree."""

    def __init__(self, e_subtree_mask, v_subtree_mask):
        """Initialize visitor with edge and vertex mask property maps.

        Parameters
        ----------
        e_subtree_mask : graph_tool.EdgePropertyMap
            Edge property map to mark edges in subtree.
        v_subtree_mask : graph_tool.VertexPropertyMap
            Vertex property map to mark vertices in subtree.
        """
        self.e_subtree_mask = e_subtree_mask
        self.v_subtree_mask = v_subtree_mask

    def tree_edge(self, e):
        self.e_subtree_mask[e] = 1

    def discover_vertex(self, u):
        self.v_subtree_mask[u] = 1


### DFS generic function wrapper


def dfsearch(
    g: Graph,
    visitor: DFSVisitor,
    init_kwargs: dict | None = None,
    init_vertex_properties: dict | None = None,
    init_edge_properties: dict | None = None,
    root: int = 0,
    bind: bool = True,
):
    """Wrapper for graph_tool's dfs_search that initializes properties and binds
    them to the graph if bind = True.

    Parameters
    ----------
    g : graph_tool.Graph
        Graph to search.
    visitor : graph_tool.DFSVisitor
        Visitor object that defines the behaviour of the search.
    init_kwargs : dict, optional
        Keyword arguments passed to the visitor constructor.
    init_vertex_properties : dict, optional
        Dictionary of vertex property name and type pairs to initialize before
        the search. For example, {"depth": "int"} will create a vertex property
        called "depth" of type int and pass it to the visitor.
    init_edge_properties : dict, optional
        Dictionary of edge property name and type pairs to initialize before
        the search.
    root : int, optional
        Index of the root vertex to start the search from, by default 0.
    bind : bool, optional
        Whether to bind the initialized properties to the graph after the search,
        by default True. If False, the properties are not bound to the graph and
        instead returned as a dictionary.

    Returns
    -------
    visitor : DFSVisitor
        The visitor instance after the search.
    dict (if bind = False and properties were initialized)
        A dictionary of property name and property pairs for all initialized
        properties. For example, {"depth": vdepth} where vdepth is a vertex
        property map containing the depth of each vertex from the root.
    """

    if init_kwargs is None:
        init_kwargs = {}
    if init_vertex_properties is None:
        init_vertex_properties = {}
    if init_edge_properties is None:
        init_edge_properties = {}

    # initialized properties
    vprops = {name: g.new_vp(ptype) for name, ptype in init_vertex_properties.items()}
    eprops = {name: g.new_ep(ptype) for name, ptype in init_edge_properties.items()}
    properties = {**vprops, **eprops}

    # initialize visitor
    vis = visitor(**init_kwargs, **properties)

    # dfs search
    dfs_search(g, root, vis)

    if bool(properties):
        # bind properties to graph or return them as a dictionary
        if bind:
            for name, prop in vprops.items():
                g.vertex_properties[name] = prop
            for name, prop in eprops.items():
                g.edge_properties[name] = prop
        # may need to change this to return visitor?
        else:
            return properties
    return vis


def df_iterator(g: Graph, root: int = 0, array: bool = True) -> Iterable | ndarray:
    """Return depth first search iterator or array of visited edges from root.

    Parameters
    ----------
    g : Graph
        Graph object.
    root : int, optional
        Root node index, by default 0.
    array : bool, optional
        If False, returns iterator over edges, otherwise returns a numpy.ndarray,
        by default True.

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in depth first search traversal order.
    """
    return dfs_iterator(g, root, array)


### DFS visitors


class PostOrderVisitor(DFSVisitor):
    """Visitor class to get post order traversal."""

    def __init__(self):
        """Initialize visitor with empty post-order list."""
        self.post_order = []

    def finish_vertex(self, v):
        self.post_order.append(int(v))


class ReduceVisitor(DFSVisitor):
    """Visitor class to obtain graph edges removing transitive vertices."""

    def __init__(self, graph, starts, stops):
        """Initialize visitor for graph reduction.

        Parameters
        ----------
        graph : graph_tool.Graph
            Input graph with 'Path_length' edge property.
        starts : array-like
            Vertex indices to start new edges from (branches and root).
        stops : array-like
            Vertex indices to end edges at (branches and leaves, excluding root).
        """
        self.graph = graph
        self.starts = starts
        self.stops = stops
        self.curr_length = 0.0
        self.edge_source = []
        self.edge_target = []
        self.path_lengths = []

    def tree_edge(self, e):
        """Edge behaviour during traversal."""
        # add length
        self.curr_length += self.graph.ep["Path_length"][e]
        # if source in starts
        if e.source() in self.starts:
            # add to starts
            self.edge_source.append(int(e.source()))
            # set current length to length of this (starting) edge
            self.curr_length = self.graph.ep["Path_length"][e]
        # if the target is in stops
        if e.target() in self.stops:
            # add to targets
            self.edge_target.append(int(e.target()))
            # add current length to path_lengths
            self.path_lengths.append(self.curr_length)


class AngleVisitor(DFSVisitor):

    def __init__(self, graph, starts, stops, x_prop, y_prop, z_prop):
        self.graph = graph
        self.starts = starts
        self.stops = stops

        self.x_prop = x_prop
        self.y_prop = y_prop
        self.z_prop = z_prop

        self._section_start = None

        self._edge_x = []
        self._edge_y = []
        self._edge_z = []

        self.edge_source = []
        self.edge_target = []
        self.mean_angles = []
        self.angle_variances = []

    def _pos(self, v):
        return (
            float(self.x_prop[v]),
            float(self.y_prop[v]),
            float(self.z_prop[v]),
        )

    def tree_edge(self, e):

        src = e.source()
        tgt = e.target()

        if src in self.starts:
            self._section_start = src
            self._edge_x.clear()
            self._edge_y.clear()
            self._edge_z.clear()

        x0, y0, z0 = self._pos(src)
        x1, y1, z1 = self._pos(tgt)

        self._edge_x.append(x1 - x0)
        self._edge_y.append(y1 - y0)
        self._edge_z.append(z1 - z0)

        if tgt in self.stops:
            self._finalise_section(tgt)

    def _finalise_section(self, stop_vertex):

        x0, y0, z0 = self._pos(self._section_start)
        x1, y1, z1 = self._pos(stop_vertex)

        section_x = x1 - x0
        section_y = y1 - y0
        section_z = z1 - z0

        edge_x = asarray(self._edge_x, dtype=float64)
        edge_y = asarray(self._edge_y, dtype=float64)
        edge_z = asarray(self._edge_z, dtype=float64)

        angles = angle(
            edge_x,
            edge_y,
            edge_z,
            section_x,
            section_y,
            section_z,
            assume_normalized=False,
        )

        self.edge_source.append(int(self._section_start))
        self.edge_target.append(int(stop_vertex))

        self.mean_angles.append(
            angular_mean(angles)
        )

        self.angle_variances.append(
            angular_var(angles)
        )

        self._section_start = None
        self._edge_x.clear()
        self._edge_y.clear()
        self._edge_z.clear()