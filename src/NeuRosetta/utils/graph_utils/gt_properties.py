"""Check and get graph property names"""

from typing import List
import itertools

from graph_tool.all import Graph

### getting properties
def _property_dict(g:Graph) -> dict:
    """Dictionary of propertiess"""
    return {"g":list(g.graph_properties.keys()),
            "v": list(g.vertex_properties.keys()),
            "e": list(g.edge_properties.keys())}

def _get_properties(g: Graph, level: str | List = "all") -> List | dict:
    """Get list of property names at given level """

    # accepted levels
    accepted_levels = ["g","v","e","all"]
    # raise if wrong value passed
    if isinstance(level, str):
        if level not in accepted_levels:
            raise AttributeError(f"level must be one of {accepted_levels}")

    # property dict
    prop_keys = _property_dict(g)

    # return what we want
    if level == "all":
        return list(itertools.chain.from_iterable(list(prop_keys.values())))
    if isinstance(level, str):
        return prop_keys[level]
    return {k:prop_keys[k] for k in level if k in accepted_levels and k != "all"}

def revert_core_properties(g):
    """Revert bound properties to core"""

    # remove unecessary properties 
    core_vps = ["coordinates","ids","node_type","radius"]
    core_gps = ["ID","metadata"]
    core_eps = ["Path_length", "Euclidean_length"]

    levels = ["e","v","g"]
    prop_dict = _get_properties(g, level = levels)

    # remove vps
    for p in prop_dict['v']:
        if p not in core_vps:
            del g.vp[p]
    # remove eps
    for p in prop_dict['e']:
        if p not in core_eps:
            del g.ep[p]
    # remove gps
    for p in prop_dict['g']:
        if p not in core_gps:
            del g.gp[p]

### checking properties

def g_has_property(g: Graph, prop: str, level: str | List = "all"):
    """Check if a graph has aproperty"""
    
    # get properties
    props = _get_properties(g, level)

    # if we passed a list to _get_properties we will have a dictionary, change this
    if isinstance(props, dict):
        props = list(itertools.chain.from_iterable(list(props.values())))
    
    return prop in props

# property missing error
class _InternalPropertyMissingError(Exception):
    """Raised when a required internal property is missing from a graph_tool.Graph object."""

    def __init__(self, missing_property, level):
        message = f"Internal Property Missing: {missing_property} at level(s) {level}"
        super().__init__(message)
        self.missing_property = missing_property

def raise_internal_property_missing(g: Graph, prop: str, level: str | List = "all") -> None:
    """raise if property is missing"""
    # get internal properties
    props = _get_properties(g, level)

    if prop not in props:
        raise _InternalPropertyMissingError(prop, level)

### Binding properties (internalising to graph)
def bind_vertex_property(g, property_name, property_dtype, property_data):
    """ Bind property to vertices"""
    g.vp[property_name] = g.new_vp(property_dtype,property_data)

def bind_edge_property(g: Graph, property_name, property_dtype, property_data):
    """ Bind property to edges"""
    g.ep[property_name] = g.new_ep(property_dtype, property_data)

def bind_graph_property(g, property_name, property_dtype, property_data):
    """Bind property to Graph"""
    g.gp[property_name] = g.new_gp(property_dtype, property_data)
