"""Helpers for :class:`~neurosetta.core.tree._Tree` graph binding and setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graph_tool.all import Graph

from .metadata import _MetadataDict, unwrap_metadata, wrap_metadata

if TYPE_CHECKING:
    from .tree import _Tree


def bind_tree_id(graph: Graph, value: int) -> None:
    """Write ``ID`` onto ``graph.gp``."""
    value = int(value)
    if "ID" not in graph.gp:
        graph.gp["ID"] = graph.new_gp("long", value)
    else:
        graph.gp["ID"] = value


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


def copy_tree_graph(graph: Graph) -> tuple[int, dict, Graph]:
    """Return ``(ID, metadata, graph)`` for a shallow tree copy."""
    copied = graph.copy()
    meta = unwrap_metadata(copied.gp["metadata"]).copy()
    return int(copied.gp["ID"]), meta, copied


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
