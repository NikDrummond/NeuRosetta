"""Topological Morphology Descriptor (TMD) on graph-tool trees."""

from __future__ import annotations

from typing import Any

import numpy as np
from graph_tool.all import Graph, VertexPropertyMap, dfs_search
from numpy.typing import NDArray

from ..graph_utils.gt_properties import bind_graph_property
from ..graph_utils.vertex_inds import root_index
from .visitors import TMDVisitor

TMDResult = dict[str, Any]
TMD_GP = "TMD"


def _resolve_func(g: Graph, func: str | VertexPropertyMap) -> tuple[str | None, VertexPropertyMap]:
    """Resolve a filtration to ``(name_or_None, VertexPropertyMap)``."""
    if isinstance(func, str):
        if func not in g.vp:
            raise KeyError(f"Vertex property {func!r} not found on graph")
        return func, g.vp[func]
    if isinstance(func, VertexPropertyMap):
        return None, func
    raise TypeError(
        f"func must be a vertex property name (str) or VertexPropertyMap; got {type(func)!r}"
    )


def normalize_tmd_payload(result: dict[str, Any]) -> TMDResult:
    """Return a pickle-safe TMD dict with contiguous numpy arrays.

    Used when binding to ``graph.gp['TMD']`` for ``.nr`` persistence.
    """
    return {
        "function": result["function"],
        "birth_ind": np.asarray(result["birth_ind"], dtype=np.int64).copy(),
        "death_ind": np.asarray(result["death_ind"], dtype=np.int64).copy(),
        "birth": np.asarray(result["birth"], dtype=np.float64).copy(),
        "death": np.asarray(result["death"], dtype=np.float64).copy(),
        "survival_len": np.asarray(result["survival_len"], dtype=np.float64).copy(),
    }


def tmd(
    g: Graph,
    func: str | VertexPropertyMap,
    bind: bool = False,
) -> TMDResult:
    """Compute the Topological Morphology Descriptor of a rooted tree.

    Uses a single DFS on the original graph (no reduction / reindexing).
    Complexity is ``O(V + E)`` = ``O(V)`` for a tree.

    Descendant leaves define active components. At each branch the component
    with the maximum leaf filtration value survives; losing components pair
    with the branch vertex. The final component pairs with the root. Original
    graph-tool vertex indices are retained in ``birth_ind`` / ``death_ind``.

    Parameters
    ----------
    g : Graph
        Rooted directed tree with edges oriented parent → child.
    func : str | VertexPropertyMap
        Filtration: vertex property name or property map.
    bind : bool, optional
        If True, store the result as graph property ``\"TMD\"`` (object) via
        :func:`~neurosetta.utils.graph_utils.bind_graph_property`. This property
        is pickled into ``.nr`` files with the graph. By default False.

    Returns
    -------
    dict
        Keys:

        - ``function``: input name (str) or ``None`` if a property map was passed
        - ``birth_ind``: ``int64`` array of branch/root vertex indices
        - ``death_ind``: ``int64`` array of descendant leaf indices
        - ``birth``: filtration values at ``birth_ind``
        - ``death``: filtration values at ``death_ind``
        - ``survival_len``: ``|death - birth|``

    Notes
    -----
    Historical NeuRosetta naming: ``birth_ind`` is the branch/root vertex and
    ``death_ind`` is the descendant leaf. For a monotonic root-distance
    function the branch/root value is normally the lower endpoint and the leaf
    value the upper endpoint; use
    :func:`~neurosetta.utils.topology.diagrams.persistence_diagram` with
    ``normalize=True`` for generic TDA software that expects
    ``birth <= death``.
    """
    root = root_index(g)
    func_name, f = _resolve_func(g, func)

    visitor = TMDVisitor(func=f, n_vertices=g.num_vertices(), root=root)
    dfs_search(g, root, visitor)

    birth_ind = np.asarray(visitor.birth_ind, dtype=np.int64)
    death_ind = np.asarray(visitor.death_ind, dtype=np.int64)
    values: NDArray[np.floating] = f.a
    birth = np.asarray(values[birth_ind], dtype=np.float64)
    death = np.asarray(values[death_ind], dtype=np.float64)
    survival_len = np.abs(death - birth)

    result = normalize_tmd_payload(
        {
            "function": func_name if func_name is not None else func,
            "birth_ind": birth_ind,
            "death_ind": death_ind,
            "birth": birth,
            "death": death,
            "survival_len": survival_len,
        }
    )

    if bind:
        bind_graph_property(g, TMD_GP, "object", result)

    return result


def get_bound_tmd(g: Graph, func: str | VertexPropertyMap | None = None) -> TMDResult | None:
    """Return a bound ``TMD`` graph property if present and compatible with ``func``.

    After ``.nr`` load the value is a dict of numpy arrays (pickle round-trip).
    Arrays are re-normalized to contiguous dtypes when retrieved.
    """
    if TMD_GP not in g.gp:
        return None
    result = g.gp[TMD_GP]
    if not isinstance(result, dict):
        return None
    try:
        result = normalize_tmd_payload(result)
    except (KeyError, TypeError, ValueError):
        return None

    if func is None:
        return result
    if isinstance(func, str):
        return result if result.get("function") == func else None
    return result if result.get("function") is func else None
