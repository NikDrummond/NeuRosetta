from graph_tool.all import Graph

### checking properties and error handling
def _check_swc_columns(df, error_type=ValueError):
    """
    Checks if required columns exist in a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame to check.
        required_columns (list): List of required column names.
        error_type (Exception): Type of exception to raise (default: ValueError).
        
    Raises:
        error_type: If any required column is missing.
    """
    required_columns = ['node_id','type','x','y','z','radius','parent_id']
    missing = [col for col in required_columns if col not in df.columns]
    
    if missing:
        raise error_type(f"Missing required columns: {missing}")
    
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
        raise InternalPropertyMissingError(property)
