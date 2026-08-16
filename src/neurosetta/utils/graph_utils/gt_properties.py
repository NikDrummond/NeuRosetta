"""Check, list, get, set, and bind graph-tool property maps."""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Literal

from graph_tool.all import Graph
from numpy import asarray, ndarray

Level = Literal["g", "v", "e"]
_LEVEL_MAP = {"g": "gp", "v": "vp", "e": "ep"}
_LEVEL_ALIASES: dict[str, Level | Literal["all"]] = {
    "g": "g",
    "graph": "g",
    "v": "v",
    "vertex": "v",
    "node": "v",
    "n": "v",
    "e": "e",
    "edge": "e",
    "all": "all",
}
_ACCEPTED_LEVEL_NAMES = tuple(_LEVEL_ALIASES)


def _normalize_level(level: str) -> Level | Literal["all"]:
    """Map short or long level names to canonical ``g`` / ``v`` / ``e`` / ``all``."""
    key = level.lower().strip()
    try:
        return _LEVEL_ALIASES[key]
    except KeyError:
        raise AttributeError(
            f"level must be one of {_ACCEPTED_LEVEL_NAMES} "
            "(short forms g/v/e/n, or graph/vertex|node/edge)"
        ) from None


def _normalize_levels(level: str | list[str]) -> str | list[str]:
    if isinstance(level, str):
        return _normalize_level(level)
    return [_normalize_level(lev) for lev in level]


### getting properties


def _property_maps(g: Graph, level: Level | str):
    """Return the property map dict for a level (gp / vp / ep)."""
    if isinstance(level, str):
        level = _normalize_level(level)
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


def _get_properties(g: Graph, level: str | list[str] = "all") -> list[str] | dict:
    """Get list of property names at given level.

    Parameters
    ----------
    g : Graph
        Graph to inspect.
    level : str | List[str], optional
        Property level to retrieve. Accepts ``"g"``/``"graph"``, ``"v"``/``"vertex"``/``"node"``,
        ``"e"``/``"edge"``, ``"all"``, or a list thereof. By default ``"all"``.

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
    level = _normalize_levels(level)
    accepted_levels = ["g", "v", "e", "all"]

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

    Core vertex properties: "x", "y", "z", "ids", "node_type", "radius"
    Core graph properties: "ID", "metadata"
    Core edge properties: "Path_length", "Euclidean_length"

    Parameters
    ----------
    g : Graph
        Graph to revert to core properties.
    """
    # remove unnecessary properties
    core_vps = ["x", "y", "z", "ids", "node_type", "radius"]
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


def g_has_property(g: Graph, prop: str, level: str | list[str] = "all") -> bool:
    """Check if a graph has a property at the specified level.

    Parameters
    ----------
    g : Graph
        Graph to check.
    prop : str
        Property name to search for.
    level : str | List[str], optional
        Property level(s) to search. Accepts ``"g"``/``"graph"``, ``"v"``/``"vertex"``/``"node"``,
        ``"e"``/``"edge"``, ``"all"``, or a list thereof. By default ``"all"``.

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

    def __init__(self, missing_property: str, level: str | list[str]):
        message = f"Internal Property Missing: {missing_property} at level(s) {level}"
        super().__init__(message)
        self.missing_property = missing_property


def raise_internal_property_missing(g: Graph, prop: str, level: str | list[str] = "all") -> None:
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


def bind_vertex_property(g: Graph, property_name: str, property_dtype: str, property_data) -> None:
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


def bind_edge_property(g: Graph, property_name: str, property_dtype: str, property_data) -> None:
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


def bind_graph_property(g: Graph, property_name: str, property_dtype: str, property_data) -> None:
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
    property_data,
) -> None:
    """Create a vector<> property via ``set_2d_array`` (layout ``(d, n)``)."""
    n = g.num_vertices() if level == "v" else g.num_edges()
    arr = _coerce_to_dims_by_n(property_data, n, name=property_name)
    if level == "v":
        prop = g.new_vp(property_dtype)
        prop.set_2d_array(arr)
        g.vp[property_name] = prop
    else:
        prop = g.new_ep(property_dtype)
        prop.set_2d_array(arr)
        g.ep[property_name] = prop


def _coerce_to_dims_by_n(data, n: int, *, name: str) -> ndarray:
    """Coerce array data to graph-tool ``set_2d_array`` layout ``(d, n)``.

    ``n`` is the number of vertices or edges at the target level.

    - ``(d, n)`` with ``d != n``: already SoA, returned as-is
    - ``(n, d)`` with ``d != n``: AoS, transposed
    - ``(n, n)``: ambiguous; warn and assume SoA (no transpose)
    """
    arr = asarray(data)
    if arr.ndim != 2:
        raise ValueError(
            f"Property {name!r}: vector<> properties require a 2d array "
            f"(d, n) or (n, d); got ndim={arr.ndim}, shape={getattr(arr, 'shape', None)}"
        )

    a, b = arr.shape
    if b == n and a != n:
        return arr  # (d, n)
    if a == n and b != n:
        return arr.T  # (n, d) → (d, n)
    if a == n and b == n:
        warnings.warn(
            f"Property {name!r}: square array shape {arr.shape} is ambiguous "
            f"(could be (d, n) or (n, d) with n={n}); "
            "assuming SoA (d, n) for set_2d_array",
            UserWarning,
            stacklevel=3,
        )
        return arr  # assume already (d, n)
    raise ValueError(
        f"Property {name!r}: array shape {arr.shape} does not match level "
        f"size n={n} on either axis (need (d, n) or (n, d))"
    )


def list_properties(g: Graph, level: str | list[str] = "all") -> list[str] | dict[str, list[str]]:
    """List property names on a graph.

    Parameters
    ----------
    g : Graph
        Graph to inspect.
    level : str | list[str], optional
        ``"all"`` → ``{"g": [...], "v": [...], "e": [...]}``.
        ``"g"``/``"graph"``, ``"v"``/``"vertex"``/``"node"``, ``"e"``/``"edge"`` → list of names.
        A list of levels → dict subset.
    """
    if level == "all":
        return _property_dict(g)
    return _get_properties(g, level)


def _property_value(
    g: Graph,
    name: str,
    level: Level,
    *,
    as_array: bool,
    SoA: bool,
) -> Any:
    """Read one property at a known level (must exist)."""
    maps = _property_maps(g, level)
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


def get_property(
    g: Graph,
    name: str,
    level: Level | None = None,
    *,
    as_array: bool = True,
    SoA: bool = False,
) -> Any:
    """Get a property map (or its array values) from a graph.

    Parameters
    ----------
    g : Graph
        Graph to read from.
    name : str
        Property name.
    level : str or None, optional
        Property level: ``"g"``/``"graph"``, ``"v"``/``"vertex"``/``"node"``, or
        ``"e"``/``"edge"``. If ``None`` (default), search all levels. A single
        match is returned directly; if the name exists at multiple levels,
        warn and return ``{level: value, ...}``.
    as_array : bool, optional
        If True, return numpy data instead of the PropertyMap.
        Scalar v/e props use ``.a``. Vector props use ``get_2d_array()``.
        Graph props return the scalar/object value (same as PropertyMap access).
    SoA : bool, optional
        Only for vector<> props with ``as_array=True``. If True, return
        shape ``(d, N)``; if False (default), return ``(N, d)``.
    """
    if level is not None:
        maps = _property_maps(g, level)
        if name not in maps:
            raise KeyError(f"Property {name!r} not found at level {level!r}")
        return _property_value(g, name, level, as_array=as_array, SoA=SoA)

    found: dict[str, Any] = {}
    for lvl in ("g", "v", "e"):
        if name in _property_maps(g, lvl):
            found[lvl] = _property_value(g, name, lvl, as_array=as_array, SoA=SoA)

    if not found:
        raise KeyError(f"Property {name!r} not found at any level (g/v/e)")

    if len(found) == 1:
        return next(iter(found.values()))

    warnings.warn(
        f"Property {name!r} found at multiple levels {list(found)}; returning dict keyed by level",
        UserWarning,
        stacklevel=2,
    )
    return found


def set_property(
    g: Graph,
    name: str,
    data: Any,
    level: Level,
    *,
    dtype: str | None = None,
    create: bool = False,
) -> None:
    """Set an existing property, or create one with ``create=True``.

    Parameters
    ----------
    g : Graph
        Graph to write to.
    name : str
        Property name.
    data : array-like or scalar
        Values to write. For vector<> props, pass a 2d array as either
        ``(d, n)`` (dimensions × vertices/edges, graph-tool SoA) or
        ``(n, d)`` (AoS); layout is inferred from ``n`` at ``level``.
        Written via ``set_2d_array``.
    level : str
        Property level: ``"g"``/``"graph"``, ``"v"``/``"vertex"``/``"node"``, or ``"e"``/``"edge"``.
    dtype : str, optional
        Required when ``create=True`` (e.g. ``"double"``, ``"vector<double>"``).
    create : bool, optional
        If True and the property is missing, bind a new one. If False and
        missing, raise ``KeyError``.
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
            if level not in ("v", "e"):
                raise ValueError("vector<> properties require level 'v' or 'e'")
            _bind_vector_property(g, level, name, dtype, data)
        elif asarray(data).ndim == 2:
            raise ValueError(
                f"Property {name!r}: 2d data requires a vector<> dtype (got dtype={dtype!r})"
            )
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
        if arr.ndim != 2:
            raise ValueError(
                f"Property {name!r} at level {level!r} is vector<>, but data is "
                f"{arr.ndim}d with shape {arr.shape}; pass a 2d array (d, n) or (n, d)"
            )
        n = g.num_vertices() if level == "v" else g.num_edges()
        prop.set_2d_array(_coerce_to_dims_by_n(data, n, name=name))
        return

    arr = asarray(data)
    if arr.ndim == 2:
        raise ValueError(
            f"Property {name!r} at level {level!r} is scalar, but data is 2d with shape {arr.shape}"
        )
    prop.a = arr


def _set_coords_prop(g, coords: ndarray):

    if coords.ndim != 2:
        raise ValueError(f"Given coordinate array must be 2 dimensional, not {coords.ndim}")

    if coords.shape[1] == 3:
        coords = coords.T
    elif coords.shape[0] != 3:
        raise ValueError("Given coordinate array must have 3 rows or columns (x,y,z)")

    set_property(g, "x", coords[0], "v")
    set_property(g, "y", coords[1], "v")
    set_property(g, "z", coords[2], "v")


def del_property(g: Graph, name: str, level: Level) -> None:
    """Delete a property map from the graph."""
    maps = _property_maps(g, level)
    if name not in maps:
        raise KeyError(f"Property {name!r} not found at level {level!r}")
    del maps[name]
