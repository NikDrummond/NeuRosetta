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
        raise AttibuteError(f"level must be None or one of {accepted_levels}")
    
    return prop in props


