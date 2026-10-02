"""Helpers for :class:`~neurosetta.core.tree._Tree` graph binding and setup."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any

from graph_tool.all import Graph

from .metadata import _MetadataDict, unwrap_metadata, wrap_metadata
from .stone_helpers import validate_stone_id

if TYPE_CHECKING:
    from .tree import _Tree


def validate_tree_id(value: Hashable) -> Hashable:
    """Validate a logical Tree ``ID`` (hashable scalar; not ``bool``)."""
    return validate_stone_id(value)


def normalize_tree_name(value: Any, *, fallback_id: Hashable | None = None) -> str:
    """Return a non-empty string artifact name.

    When *value* is ``None``, fall back to ``str(fallback_id)``. Empty strings
    are rejected.
    """
    if value is None:
        if fallback_id is None:
            raise TypeError("tree name is None and no fallback ID was provided")
        return str(fallback_id)
    name = str(value)
    if not name:
        raise ValueError("tree name must be a non-empty string")
    return name


def _rebind_object_gp(graph: Graph, key: str, value: Any) -> None:
    """Write an object graph property, replacing any prior typed property."""
    if key in graph.gp:
        del graph.gp[key]
    graph.gp[key] = graph.new_gp("object", value)


def bind_tree_id(graph: Graph, value: Hashable) -> None:
    """Write logical ``ID`` onto ``graph.gp`` as an object property.

    Accepts ``int`` and ``str`` (and other hashable scalars). Does **not**
    coerce digit-strings to integers — ``\"00123\"`` stays ``\"00123\"``.
    """
    value = validate_tree_id(value)
    _rebind_object_gp(graph, "ID", value)


def bind_tree_name(graph: Graph, value: str) -> None:
    """Write artifact ``name`` onto ``graph.gp`` as an object property."""
    name = normalize_tree_name(value)
    _rebind_object_gp(graph, "name", name)


def tree_id_from_graph(graph: Graph) -> Hashable:
    """Read logical ``ID`` from ``graph.gp`` without lossy type coercion.

    Legacy ``long`` / integer graph-tool properties are returned as Python
    ``int``. Object / string properties are returned as stored (digit-strings
    stay strings).
    """
    if "ID" not in graph.gp:
        raise KeyError("graph is missing gp['ID']")
    raw = graph.gp["ID"]
    if isinstance(raw, bool):
        raise TypeError("Stone ID must not be a bool")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, int):
        return int(raw)  # normalise numpy/python ints
    # Legacy long PropertyMap / numpy integer — never apply int() to str.
    try:
        return int(raw)
    except (TypeError, ValueError):
        return validate_tree_id(raw)  # type: ignore[arg-type]


def tree_name_from_graph(graph: Graph, *, fallback_id: Hashable | None = None) -> str:
    """Read artifact ``name`` from ``graph.gp``, or fall back to ``str(ID)``."""
    if "name" in graph.gp:
        raw = graph.gp["name"]
        if raw is not None and str(raw):
            return normalize_tree_name(raw)
    if fallback_id is None:
        fallback_id = tree_id_from_graph(graph)
    return normalize_tree_name(None, fallback_id=fallback_id)


def copy_tree_identity(src: Graph, dst: Graph) -> None:
    """Copy logical ``ID`` and artifact ``name`` from *src* onto *dst*."""
    tree_id = tree_id_from_graph(src)
    bind_tree_id(dst, tree_id)
    bind_tree_name(dst, tree_name_from_graph(src, fallback_id=tree_id))


def bind_tree_metadata(graph: Graph, value: dict | _MetadataDict) -> None:
    """Write wrapped metadata onto ``graph.gp``."""
    if not isinstance(value, (dict, _MetadataDict)):
        raise TypeError("metadata must be a dict")
    wrapped = wrap_metadata(value)
    wrapped.setdefault("Flag", False)
    if "metadata" not in graph.gp:
        graph.gp["metadata"] = graph.new_gp("object", wrapped)
    else:
        graph.gp["metadata"] = wrapped


def metadata_from_graph(graph: Graph) -> dict:
    """Extract a plain metadata dict from a graph, migrating legacy ``flag`` gp."""
    meta = dict(unwrap_metadata(graph.gp["metadata"]))
    if "flag" in graph.gp:
        meta.setdefault("Flag", bool(graph.gp["flag"]))
        del graph.gp["flag"]
    meta.setdefault("Flag", False)
    return meta


def copy_tree_graph(graph: Graph) -> tuple[Hashable, str, dict, Graph]:
    """Return ``(ID, name, metadata, graph)`` for a shallow tree copy.

    Attached :class:`~neurosetta.core.synapses.Synapses` (if any) are deep-copied
    so edits on the clone do not mutate the original table. Attached neuron
    meshes are likewise cloned (live ``Tree_mesh`` or verts/faces payload).
    """
    copied = graph.copy()
    meta = unwrap_metadata(copied.gp["metadata"]).copy()
    if "synapses" in copied.gp and copied.gp["synapses"] is not None:
        syn = copied.gp["synapses"]
        copied.gp["synapses"] = syn.copy()
    if "mesh" in copied.gp and copied.gp["mesh"] is not None:
        from ..ops.tree_graphs.tree_mesh import _copy_mesh_value

        copied.gp["mesh"] = _copy_mesh_value(copied.gp["mesh"])
    tree_id = tree_id_from_graph(copied)
    name = tree_name_from_graph(copied, fallback_id=tree_id)
    # Migrate legacy long ID → object gps on the copy.
    bind_tree_id(copied, tree_id)
    bind_tree_name(copied, name)
    return tree_id, name, meta, copied


def ensure_edge_lengths(tree: _Tree) -> None:
    """Bind ``Path_length`` or ``Euclidean_length`` when not already present.

    Reduced trees may carry ``Path_length`` from the full reconstruction; in
    that case lengths are left unchanged. Graphs without coordinates are skipped
    (not yet valid neuron trees).
    """
    from ..ops.tree_graphs.tree_checks import has_property
    from ..ops.tree_graphs.tree_path_lengths import get_edge_length
    from ..utils.graph_utils import g_has_property

    if has_property(tree, "Path_length", "e") or has_property(tree, "Euclidean_length", "e"):
        return
    if not all(g_has_property(tree.graph, c, "v") for c in ("x", "y", "z")):
        return
    get_edge_length(tree, bind=True)
