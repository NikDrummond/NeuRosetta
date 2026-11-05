"""Resolve numeric edge weights for path distances.

``ep=None`` selects ``Path_length``, then ``Euclidean_length``. An explicit
name or :class:`~graph_tool.EdgePropertyMap` is used as given. Missing
explicit properties are not replaced, and lengths are not invented from
coordinates. Reduced trees can store a curved ``Path_length`` that differs
from the straight ``Euclidean_length`` of the same edge.
"""

from __future__ import annotations

from graph_tool import EdgePropertyMap
from graph_tool.all import Graph
from numpy import any as np_any
from numpy import array, float64, isfinite, issubdtype, number

_FALLBACK = ("Path_length", "Euclidean_length")


def resolve_edge_property(
    g: Graph,
    ep: str | EdgePropertyMap | None,
    *,
    nonnegative: bool = True,
) -> EdgePropertyMap:
    """Return the edge property map used as path weights.

    The graph is not modified. The returned map is the existing property,
    not a copy.

    Parameters
    ----------
    g : Graph
        Directed tree whose edge properties are read.
    ep : str or EdgePropertyMap or None
        ``None`` selects ``Path_length`` if it exists, otherwise
        ``Euclidean_length``. A string names one property. A property map
        is used directly.
    nonnegative : bool, optional
        Reject negative values. By default True. Zero is allowed.

    Returns
    -------
    EdgePropertyMap
        Scalar numeric weights, one value per edge.

    Raises
    ------
    ValueError
        If ``ep`` is None and neither standard length property exists, if an
        explicit name is missing, or if a value is non-finite or negative.
    TypeError
        If the property is not a numeric scalar (strings, vectors, booleans).
    """
    if ep is None:
        for name in _FALLBACK:
            if name in g.ep:
                ep = name
                break
        else:
            raise ValueError(
                "No edge length property found. Pass ep explicitly, or add "
                "Path_length or Euclidean_length. Lengths are not computed "
                "from coordinates."
            )

    if isinstance(ep, str):
        if ep not in g.ep:
            raise ValueError(
                f"Edge property {ep!r} is not on the graph. "
                "An explicit property is not replaced with Path_length "
                "or Euclidean_length."
            )
        prop = g.ep[ep]
    elif isinstance(ep, EdgePropertyMap):
        prop = ep
    else:
        raise TypeError(
            "ep must be None, an edge-property name, or an EdgePropertyMap, "
            f"got {type(ep).__name__}."
        )

    _validate_weights(prop, nonnegative=nonnegative)
    return prop


def as_double_weights(g: Graph, prop: EdgePropertyMap) -> EdgePropertyMap:
    """Return a double-valued weight map for ``shortest_distance``.

    Integer weights make unreachable vertices the integer maximum rather
    than infinity. A temporary double map is created only when ``prop`` is
    not already ``double``. It is not inserted under a name in ``g.ep``.

    Parameters
    ----------
    g : Graph
        Graph that owns ``prop``.
    prop : EdgePropertyMap
        Validated scalar weights.

    Returns
    -------
    EdgePropertyMap
        Double weights. The original named property is unchanged.
    """
    if str(prop.value_type()) == "double":
        return prop
    values = array(prop.a, dtype=float64, copy=True)
    temporary = g.new_ep("double")
    temporary.a = values
    return temporary


def _validate_weights(prop: EdgePropertyMap, *, nonnegative: bool) -> None:
    try:
        values = array(prop.a, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError("Edge weights must be a numeric scalar property.") from exc
    if values.ndim != 1 or values.dtype == object or not issubdtype(values.dtype, number):
        raise TypeError(
            "Edge weights must be a numeric scalar property. "
            f"Got dtype {values.dtype} with shape {values.shape}."
        )
    if not isfinite(values).all():
        raise ValueError("Edge weights must be finite. NaN and infinity are rejected.")
    if nonnegative and np_any(values < 0):
        raise ValueError(
            "Edge weights must be nonnegative. Signed weights are not used as path lengths."
        )
