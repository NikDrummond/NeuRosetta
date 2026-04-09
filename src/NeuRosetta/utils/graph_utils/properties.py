from graph_tool.all import Graph

def g_has_property(g: Graph, prop:str, level: str | None = None) -> bool:
    """ Bool check if graph has a specific property """

    accepted_levels = ['g','v','e']

    if level == 'g':
        props = list(g.graph_properties.keys())
    elif level == 'v':
        props = list(g.vertex_properties.keys())
    elif level == 'e':
        props = list(g.edge_properties.keys())
    elif level == None:
        props = list(g.graph_properties.keys()) + list(g.vertex_properties.keys()) + list(g.edge_properties.keys())
    else:
        raise AttributeError(f"level must be None or one of {accepted_levels}")
    
    return prop in props

def _bind_vertex_property(tree, property_name: str, property_dtype: str, property_data):
    """ Bind property to vertices"""
    tree.graph.vp[property_name] = tree.graph.new_vp(property_dtype,property_data)

def _bind_edge_property(tree, property_name, property_dtype, property_data):
    """ Bind property to edges"""
    tree.graph.ep[property_name] = tree.graph.new_ep(property_dtype, property_data)
