from graph_tool.all import Graph

### checking properties and error handling

    
class _InternalPropertyMissingError(Exception):
    """Raised when a required internal property is missing from a graph_tool.Graph object."""
    def __init__(self, missing_propertie):
        message = f"Internal Property Missing: {missing_propertie}"
        super().__init__(message)
        self.missing_propertie = missing_propertie

def _check_internal_property(g:Graph, property: str) -> None:

    # get internal properties
    graph_props = list(g.graph_properties.keys())
    vertex_props = list(g.vertex_properties.keys())
    edge_props = list(g.edge_properties.keys())

    if property not in graph_props + vertex_props + edge_props:
        raise _InternalPropertyMissingError(property)
