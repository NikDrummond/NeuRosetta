"""Module for breadth and depth first traversals on graphs"""

from typing import Iterable
from numpy import ndarray

from graph_tool.all import (
    BFSVisitor,
    bfs_search,
    DFSVisitor,
    dfs_search,
    Graph,
    bfs_iterator,
    dfs_iterator,
)

### BFS taversals


def bfsearch(
    g: Graph,
    visitor: BFSVisitor,
    init_kwargs: dict | None = None,
    init_properties: dict | None = None,
    root: int = 0,
    bind: bool = True,
):
    """Wrapper for graph_tool's bfs_search that initialises properties and binds them to the graph if bind = True.

    Parameters
    ----------
    g : graph_tool.Graph
        graph to search
    visitor : graph_tool.BFSVisitor
        visitor object that defines the behaviour of the search
    init_properties : dict
        dictionary of property name and type pairs to initialise before the search.
        For example, {"depth": "int"} will create a vertex property called "vdepth" of type int and pass it to the visitor.
    root : int, optional
        index of the root vertex to start the search from, by default 0.
    bind : bool, optional
        whether to bind the initialised properties to the graph after the search, by default True.
        If False, the properties are not bound to the graph and instead returned as a dictionary.

    Returns
    -------
    dict (if bind = False)
        A dictionary of property name and property pairs for all initialised properties.
        For example, {"vdepth": vdepth} where vdepth is a vertex property map containing the depth of each vertex from the root.
    """

    if init_kwargs is None:
        init_kwargs = {}
    if init_properties is None:
        init_properties = {}

    # initialised properties
    properties = {name: g.new_vp(ptype) for name, ptype in init_properties.items()}

    vis = visitor(**init_kwargs, **properties)
    # bfs search
    bfs_search(g, root, vis)

    # bind properties to graph or return them as a dictionary
    if bool(properties):
        if bind:
            for name, prop in properties.items():
                g.vertex_properties[name] = prop
        else:
            return properties
    return vis


def bf_iterator(g: Graph, root: int = 0, array: bool = True) -> Iterable | ndarray:
    """return breadth first search iterator or array of visitied edges from root

    Parameters
    ----------
    g : Graph
        Graph object
    root : int, optional
        root node index, by default 0
    array : bool, optional
        If False, returns iterator over edges, otherwise returns a numpy.ndarray, by default True

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in breadth first search traversal order
    """
    return bfs_iterator(g, root, array)


### BFS visitors


class TreeDepthVisitor(BFSVisitor):
    """Visitor class to obtain node depths in tree"""
    def __init__(self, depth):
        self.depth = depth

    def tree_edge(self, e):
        self.depth[e.target()] = self.depth[e.source()] + 1


### DFS generic function wrapper


def dfsearch(
    g: Graph,
    visitor: DFSVisitor,
    init_kwargs: dict | None = None,
    init_properties: dict | None = None,
    root: int = 0,
    bind=True,
):
    """Wrapper for graph_tool's dfs_search that initialises properties and binds them to the graph if
    bind = True.

    Parameters
    ----------
    g : graph_tool.Graph
        graph to search
    visitor : graph_tool.DFSVisitor
        visitor object that defines the behaviour of the search
    init_properties : dict
        dictionary of property name and type pairs to initialise before the search.
        For example, {"depth": "int"} will create a vertex property called "vdepth" of type int and pass it to the
        visitor.
    root : int, optional
        index of the root vertex to start the search from, by default 0.
    bind : bool, optional
        whether to bind the initialised properties to the graph after the search, by default True.
        If False, the properties are not bound to the graph and instead returned as a dictionary.

    Returns
    -------
    dict (if bind = False)
        A dictionary of property name and property pairs for all initialised properties.
        For example, {"vdepth": vdepth} where vdepth is a vertex property map containing the depth of each
        vertex from the root.
    """

    if init_kwargs is None:
        init_kwargs = {}
    if init_properties is None:
        init_properties = {}

    # initialised properties
    properties = {name: g.new_vp(ptype) for name, ptype in init_properties.items()}

    # initialise visitor
    vis = visitor(**init_kwargs, **properties)

    # dfs search
    dfs_search(g, root, vis)

    if bool(properties):
        # bind properties to graph or return them as a dictionary
        if bind:
            for name, prop in properties.items():
                g.vertex_properties[name] = prop
        # may need to change this to return visitor?
        else:
            return properties
    return vis


def df_iterator(g: Graph, root: int = 0, array: bool = True) -> Iterable | ndarray:
    """return depth first search iterator or array of visitied edges from root

    Parameters
    ----------
    g : Graph
        Graph object
    root : int, optional
        root node index, by default 0
    array : bool, optional
        If False, returns iterator over edges, otherwise returns a numpy.ndarray, by default True

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in depth first search traversal order
    """
    return dfs_iterator(g, root, array)


### DFS visitors


class PostOrderVisitor(DFSVisitor):
    """Visitor class to get post order traversal"""
    def __init__(self):
        self.post_order = []

    def finish_vertex(self, v):
        self.post_order.append(int(v))


class ReduceVisitor(DFSVisitor):
    """Visitor class to obtain graph edges removing transitive vertices"""
    def __init__(self, graph, starts, stops):
        self.graph = graph
        self.starts = starts
        self.stops = stops
        self.curr_length = 0.0
        self.edge_source = []
        self.edge_target = []
        self.path_lengths = []

    # what to do at each edge
    def tree_edge(self, e):
        """edge behaviour during traversal"""
        # add length
        self.curr_length += self.graph.ep["Path_length"][e]
        # if sourse in starts
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
