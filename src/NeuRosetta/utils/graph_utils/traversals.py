from graph_tool.all import BFSVisitor, bfs_search, DFSVisitor, dfs_search, Graph

### BFS taversals - generic function wrrapper


def _BF_search(g: Graph, visitor: BFSVisitor, init_kwargs: dict = {}, init_properties: dict = {}, root: int = 0, bind: bool = True):
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


### BFS visitors
class Tree_depth(BFSVisitor):

    def __init__(self, depth):
        self.depth = depth

    def tree_edge(self, e):
        self.depth[e.target()] = self.depth[e.source()] + 1


### DFS generic function wrapper


def _DF_search(g: Graph, visitor: DFSVisitor, init_kwargs:dict = {}, init_properties: dict = {}, root: int = 0, bind=True):
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


### DFS visitors


class PostOrderVisitor(DFSVisitor):
    def __init__(self):
        self.post_order = []

    def finish_vertex(self, v):
        self.post_order.append(int(v))

class ReduceVisitor(DFSVisitor):
    
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
        # add length
        self.curr_length += self.graph.ep['Path_length'][e]
        # if sourse in starts 
        if e.source() in self.starts:
            # add to starts
            self.edge_source.append(int(e.source()))
            # set current length back to 0
            self.curr_length = 0.0
        # if the target is in stops 
        if e.target() in self.stops:
            # add to targets
            self.edge_target.append(int(e.target()))
            # add current length to path_lengths
            self.path_lengths.append(self.curr_length)