"""Check, list, get, set, and bind graph-tool property maps."""
from __future__ import annotations

from typing import List, Any, Literal
import itertools

from numpy import asarray, ndarray
from graph_tool.all import Graph

Level = Literal["g", "v", "e"]
_LEVEL_MAP = {"g": "gp", "v": "vp", "e": "ep"}


### getting properties


def _property_maps(g: Graph, level: Level):
    """Return the property map dict for a level (gp / vp / ep)."""
    if level not in _LEVEL_MAP:
        raise AttributeError(f"level must be one of {list(_LEVEL_MAP)}")
    return getattr(g, _LEVEL_MAP[level])


def _is_vector_property(prop) -> bool:
    """True if PropertyMap value type is vector<...>."""
    return str(prop.value_type()).startswith("vector")


def _property_dict(g: Graph) -> dict:
    """Return dictionary of property names by level.

    Parameters
    ----------
    g : Graph
        Graph to inspect.

    Returns
    -------
    dict
        Dictionary with keys 'g', 'v', 'e' mapping to lists of graph, vertex,
        and edge property names respectively.
    """
    return {
        "g": list(g.graph_properties.keys()),
        "v": list(g.vertex_properties.keys()),
        "e": list(g.edge_properties.keys()),
    }


def _get_properties(g: Graph, level: str | List[str] = "all") -> List[str] | dict:
    """Get list of property names at given level.

    Parameters
    ----------
    g : Graph
        Graph to inspect.
    level : str | List[str], optional
        Property level to retrieve. Can be "g" (graph), "v" (vertex), "e" (edge),
        "all" (all levels), or a list of levels. By default "all".

    Returns
    -------
    List[str] | dict
        If level is a string, returns list of property names at that level.
        If level is a list, returns dict mapping level to property names.

    Raises
    ------
    AttributeError
        If level is a string not in {"g", "v", "e", "all"}.
    """
    # accepted levels
    accepted_levels = ["g", "v", "e", "all"]
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
    return {k: prop_keys[k] for k in level if k in accepted_levels and k != "all"}


def revert_core_properties(g: Graph) -> None:
    """Remove non-core properties from a graph, keeping only essential ones.

    Core vertex properties: "coordinates", "ids", "node_type", "radius"
    Core graph properties: "ID", "metadata"
    Core edge properties: "Path_length", "Euclidean_length"

    Parameters
    ----------
    g : Graph
        Graph to revert to core properties.
    """
    # remove unnecessary properties
    core_vps = ["coordinates", "ids", "node_type", "radius"]
    core_gps = ["ID", "metadata"]
    core_eps = ["Path_length", "Euclidean_length"]

    levels = ["e", "v", "g"]
    prop_dict = _get_properties(g, level=levels)

    # remove vps
    for p in prop_dict["v"]:
        if p not in core_vps:
            del g.vp[p]
    # remove eps
    for p in prop_dict["e"]:
        if p not in core_eps:
            del g.ep[p]
    # remove gps
    for p in prop_dict["g"]:
        if p not in core_gps:
            del g.gp[p]


### checking properties


def g_has_property(g: Graph, prop: str, level: str | List[str] = "all") -> bool:
    """Check if a graph has a property at the specified level.

    Parameters
    ----------
    g : Graph
        Graph to check.
    prop : str
        Property name to search for.
    level : str | List[str], optional
        Property level(s) to search. Can be "g", "v", "e", "all", or list
        thereof. By default "all".

    Returns
    -------
    bool
        True if property exists at the specified level(s), False otherwise.
    """
    # get properties
    props = _get_properties(g, level)

    # if we passed a list to _get_properties we will have a dictionary, change this
    if isinstance(props, dict):
        props = list(itertools.chain.from_iterable(list(props.values())))

    return prop in props


# property missing error
class _InternalPropertyMissingError(Exception):
    """Raised when a required internal property is missing from a
    graph_tool.Graph object."""

    def __init__(self, missing_property: str, level: str | List[str]):
        message = f"Internal Property Missing: {missing_property} at level(s) {level}"
        super().__init__(message)
        self.missing_property = missing_property


def raise_internal_property_missing(
    g: Graph, prop: str, level: str | List[str] = "all"
) -> None:
    """Raise an exception if a required property is missing from the graph.

    Parameters
    ----------
    g : Graph
        Graph to check.
    prop : str
        Required property name.
    level : str | List[str], optional
        Property level(s) to check. By default "all".

    Raises
    ------
    _InternalPropertyMissingError
        If the property is not found at the specified level(s).
    """
    # get internal properties
    props = _get_properties(g, level)

    if prop not in props:
        raise _InternalPropertyMissingError(prop, level)


### Binding properties (internalising to graph)


def bind_vertex_property(
    g: Graph, property_name: str, property_dtype: str, property_data
) -> None:
    """Bind a property array to graph vertices.

    Parameters
    ----------
    g : Graph
        Graph to bind property to.
    property_name : str
        Name of the vertex property.
    property_dtype : str
        Data type string for the property (e.g., "double", "int", "vector<double>").
    property_data : array-like
        Data array to bind, must have length equal to number of vertices.
    """
    g.vp[property_name] = g.new_vp(property_dtype, property_data)


def bind_edge_property(
    g: Graph, property_name: str, property_dtype: str, property_data
) -> None:
    """Bind a property array to graph edges.

    Parameters
    ----------
    g : Graph
        Graph to bind property to.
    property_name : str
        Name of the edge property.
    property_dtype : str
        Data type string for the property (e.g., "double", "int").
    property_data : array-like
        Data array to bind, must have length equal to number of edges.
    """
    g.ep[property_name] = g.new_ep(property_dtype, property_data)


def bind_graph_property(
    g: Graph, property_name: str, property_dtype: str, property_data
) -> None:
    """Bind a property value to the graph.

    Parameters
    ----------
    g : Graph
        Graph to bind property to.
    property_name : str
        Name of the graph property.
    property_dtype : str
        Data type string for the property (e.g., "double", "int", "object").
    property_data : object
        Value to bind as the graph property.
    """
    g.gp[property_name] = g.new_gp(property_dtype, property_data)


def _bind_vector_property(
    g: Graph,
    level: Literal["v", "e"],
    property_name: str,
    property_dtype: str,
    property_data: ndarray,
    *,
    SoA: bool,
) -> None:
    """Create a vector<> property via ``set_2d_array``.

    ``property_data`` is (N, d) if ``SoA=False``, or (d, N) if ``SoA=True``.
    """
    data = asarray(property_data)
    arr = data if SoA else data.T
    if level == "v":
        prop = g.new_vp(property_dtype)
        prop.set_2d_array(arr)
        g.vp[property_name] = prop
    else:
        prop = g.new_ep(property_dtype)
        prop.set_2d_array(arr)
        g.ep[property_name] = prop


def list_properties(
    g: Graph, level: str | List[str] = "all"
) -> list[str] | dict[str, list[str]]:
    """List property names on a graph.

    Parameters
    ----------
    g : Graph
        Graph to inspect.
    level : str | list[str], optional
        ``"all"`` → ``{"g": [...], "v": [...], "e": [...]}``.
        ``"g"`` / ``"v"`` / ``"e"`` → list of names at that level.
        A list of levels → dict subset.
    """
    if level == "all":
        return _property_dict(g)
    return _get_properties(g, level)


def get_property(
    g: Graph,
    name: str,
    level: Level,
    *,
    as_array: bool = False,
    SoA: bool = False,
) -> Any:
    """Get a property map (or its array values) from a graph.

    Parameters
    ----------
    g : Graph
        Graph to read from.
    name : str
        Property name.
    level : {"g", "v", "e"}
        Property level.
    as_array : bool, optional
        If True, return numpy data instead of the PropertyMap.
        Scalar v/e props use ``.a``. Vector props use ``get_2d_array()``.
        Graph props return the scalar/object value (same as PropertyMap access).
    SoA : bool, optional
        Only for vector<> props with ``as_array=True``. If True, return
        shape ``(d, N)``; if False (default), return ``(N, d)``.
    """
    maps = _property_maps(g, level)
    if name not in maps:
        raise KeyError(f"Property {name!r} not found at level {level!r}")

    prop = maps[name]
    if not as_array:
        return prop

    # graph-level: gp access already yields the stored value
    if level == "g":
        return prop

    if _is_vector_property(prop):
        arr = prop.get_2d_array()  # (d, N)
        return arr if SoA else arr.T

    return prop.a


def set_property(
    g: Graph,
    name: str,
    data: Any,
    level: Level,
    *,
    dtype: str | None = None,
    create: bool = False,
    SoA: bool = False,
) -> None:
    """Set an existing property, or create one with ``create=True``.

    Parameters
    ----------
    g : Graph
        Graph to write to.
    name : str
        Property name.
    data : array-like or scalar
        Values to write. For vector<> props: ``(N, d)`` if ``SoA=False``,
        or ``(d, N)`` if ``SoA=True``. Written via ``set_2d_array``.
    level : {"g", "v", "e"}
        Property level.
    dtype : str, optional
        Required when ``create=True`` (e.g. ``"double"``, ``"vector<double>"``).
    create : bool, optional
        If True and the property is missing, bind a new one. If False and
        missing, raise ``KeyError``.
    SoA : bool, optional
        Layout of ``data`` for vector<> properties.
    """
    maps = _property_maps(g, level)
    exists = name in maps

    if not exists and not create:
        raise KeyError(
            f"Property {name!r} not found at level {level!r}; "
            "pass create=True (and dtype=...) to bind a new property"
        )

    if not exists:
        if dtype is None:
            raise ValueError("dtype is required when create=True")
        if level == "g":
            bind_graph_property(g, name, dtype, data)
        elif dtype.startswith("vector"):
            _bind_vector_property(g, level, name, dtype, asarray(data), SoA=SoA)
        elif level == "v":
            bind_vertex_property(g, name, dtype, data)
        else:
            bind_edge_property(g, name, dtype, data)
        return

    # --- update existing ---
    if level == "g":
        g.gp[name] = data
        return

    prop = maps[name]
    if _is_vector_property(prop):
        arr = asarray(data)
        prop.set_2d_array(arr if SoA else arr.T)
        return

    prop.a = asarray(data)


def del_property(g: Graph, name: str, level: Level) -> None:
    """Delete a property map from the graph."""
    maps = _property_maps(g, level)
    if name not in maps:
        raise KeyError(f"Property {name!r} not found at level {level!r}")
    del maps[name]