"""Tree-level synapse attachment, mapping, and basic analyses."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from graph_tool.topology import shortest_distance

from ...core import _Tree
from ...core.synapses import (
    MAPPING_VERSION,
    Synapses,
    resolve_synapse_type_filter,
)
from ...utils.geometry_utils.segments import project_points_to_segments
from ...utils.graph_utils import g_has_property
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_coordinates import get_edge_coordinates, get_root_coordinate
from .tree_path_lengths import get_edge_length, get_total_cable_length
from .tree_vertex_inds import get_edge_indices, get_root_index

SynapseTypeArg = Literal["pre", "post", "both"] | str

_SYNAPSES_GP = "synapses"


def _bind_synapses_gp(tree: _Tree, synapses: Synapses | None) -> None:
    g = tree.graph
    if synapses is None:
        if _SYNAPSES_GP in g.gp:
            del g.gp[_SYNAPSES_GP]
        return
    if _SYNAPSES_GP not in g.gp:
        g.gp[_SYNAPSES_GP] = g.new_gp("object", synapses)
    else:
        g.gp[_SYNAPSES_GP] = synapses


def get_synapses(tree: _Tree) -> Synapses | None:
    """Return attached synapses, or ``None`` if absent."""
    if not g_has_property(tree.graph, _SYNAPSES_GP, "g"):
        return None
    return tree.graph.gp[_SYNAPSES_GP]


def has_synapses(tree: _Tree) -> bool:
    """Return True when a non-empty synapse table is attached."""
    syn = get_synapses(tree)
    return syn is not None and len(syn) > 0


def invalidate_synapse_mapping(tree: _Tree) -> None:
    """Clear derived mapping fields after topology-changing edits."""
    syn = get_synapses(tree)
    if syn is None:
        return
    syn.clear_mapping()
    _bind_synapses_gp(tree, syn)


def _require_synapses(tree: _Tree) -> Synapses:
    syn = get_synapses(tree)
    if syn is None:
        raise ValueError("Tree has no synapses; call tree.set_synapses(...) first")
    return syn


def _coerce_synapses(
    data: Any,
    *,
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z"),
    type_column: str = "type",
    partner_column: str = "partner_id",
    id_column: str = "synapse_id",
) -> Synapses:
    if isinstance(data, Synapses):
        return data.copy()

    if isinstance(data, pd.DataFrame):
        df = data.copy()
        rename = {}
        cx, cy, cz = coordinate_columns
        if (cx, cy, cz) != ("x", "y", "z"):
            rename[cx] = "x"
            rename[cy] = "y"
            rename[cz] = "z"
        if type_column != "type":
            rename[type_column] = "type"
        if partner_column != "partner_id":
            rename[partner_column] = "partner_id"
        if id_column in df.columns and id_column != "synapse_id":
            rename[id_column] = "synapse_id"
        if rename:
            df = df.rename(columns=rename)
        if "synapse_id" not in df.columns:
            df.insert(0, "synapse_id", np.arange(len(df)))
        if "partner_id" not in df.columns:
            df["partner_id"] = None
        return Synapses(df)

    if isinstance(data, dict):
        return Synapses(data)

    raise TypeError(
        "Unsupported synapse input; expected Synapses, DataFrame, mapping, "
        f"or use synapses_from_arrays(...); got {type(data)!r}"
    )


def set_synapses(
    tree: _Tree,
    data: Any,
    *,
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z"),
    type_column: str = "type",
    partner_column: str = "partner_id",
    id_column: str = "synapse_id",
) -> Synapses:
    """Attach (replace) synapses on *tree*.

    Accepts a :class:`~neurosetta.core.synapses.Synapses` instance, a pandas
    DataFrame, or a column mapping. Raw coordinates are preserved; any previous
    mapping is discarded.
    """
    syn = _coerce_synapses(
        data,
        coordinate_columns=coordinate_columns,
        type_column=type_column,
        partner_column=partner_column,
        id_column=id_column,
    )
    syn.clear_mapping()
    _bind_synapses_gp(tree, syn)
    return syn


def add_synapses(
    tree: _Tree,
    data: Any,
    *,
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z"),
    type_column: str = "type",
    partner_column: str = "partner_id",
    id_column: str = "synapse_id",
) -> Synapses:
    """Append synapses to any existing table (invalidates mapping)."""
    new = _coerce_synapses(
        data,
        coordinate_columns=coordinate_columns,
        type_column=type_column,
        partner_column=partner_column,
        id_column=id_column,
    )
    existing = get_synapses(tree)
    if existing is None or len(existing) == 0:
        return set_synapses(tree, new)

    combined = pd.concat(
        [existing.to_dataframe(copy=False), new.to_dataframe(copy=False)],
        ignore_index=True,
        sort=False,
    )
    # Drop stale mapping columns before re-wrapping.
    from ...core.synapses import MAPPING_COLUMNS

    drop = [c for c in MAPPING_COLUMNS if c in combined.columns]
    if drop:
        combined = combined.drop(columns=drop)
    return set_synapses(tree, combined)


def clear_synapses(tree: _Tree) -> None:
    """Remove all attached synapses."""
    _bind_synapses_gp(tree, None)


def _drop_unmapped_synapses(syn: Synapses) -> Synapses:
    """Return a copy keeping only rows with ``mapped=True`` (empty table ok)."""
    if not syn.is_mapped:
        return syn
    kept = syn.filter(mapped=True)
    meta = dict(syn.mapping_meta)
    meta["valid"] = True
    kept._mapping_meta = meta
    return kept


def map_synapses(
    tree: _Tree,
    *,
    max_distance: float | None = None,
    drop_unmapped: bool = False,
    force: bool = False,
    method: str = "bruteforce",
    k_candidates: int = 64,
) -> Synapses:
    """Map each synapse onto the nearest morphology edge.

    Stores ``edge_index``, ``edge_fraction``, ``nearest_*``, ``distance_to_tree``,
    and ``mapped``. Synapses beyond *max_distance* are flagged ``mapped=False``;
    pass ``drop_unmapped=True`` to remove those rows after mapping.

    Parameters
    ----------
    max_distance
        Synapses farther than this from the morphology are marked unmapped
        (and dropped when *drop_unmapped* is True). ``None`` maps all.
    drop_unmapped
        If True, delete rows with ``mapped=False`` after (re)mapping. By default
        False (retain for QC / ``show_synapse_mapping``).
    method
        Passed to :func:`~neurosetta.utils.geometry_utils.segments.project_points_to_segments`.
        Use ``\"bruteforce\"`` for a guaranteed exhaustive search.
    """
    syn = _require_synapses(tree)
    if syn.mapping_valid and not force:
        if max_distance is None or syn.mapping_meta.get("max_distance") == max_distance:
            if drop_unmapped:
                syn = _drop_unmapped_synapses(syn)
                _bind_synapses_gp(tree, syn)
            return syn
        # Re-evaluate mapped flags under a new distance threshold without reprojecting.
        dist = syn.to_dataframe(copy=False)["distance_to_tree"].to_numpy(dtype=np.float64)
        mapped = np.isfinite(dist) & (dist <= float(max_distance))
        df = syn.to_dataframe(copy=True)
        df["mapped"] = mapped
        syn = Synapses(df, copy=False, mapping_meta=dict(syn.mapping_meta))
        syn._mapping_meta["max_distance"] = max_distance
        syn._mapping_meta["valid"] = True
        if drop_unmapped:
            syn = _drop_unmapped_synapses(syn)
        _bind_synapses_gp(tree, syn)
        return syn

    starts, ends = get_edge_coordinates(tree, SoA=False)
    edge_index, nearest, distance, t = project_points_to_segments(
        syn.coordinates,
        starts,
        ends,
        method=method,
        k_candidates=k_candidates,
    )

    if max_distance is None:
        mapped = np.ones(len(syn), dtype=bool)
    else:
        mapped = distance <= float(max_distance)

    lengths = np.asarray(get_edge_length(tree, bind=False), dtype=np.float64)
    distance_along = t * lengths[edge_index]

    # Unmapped rows keep nearest geometry for QC but are flagged.
    syn.set_mapping(
        edge_index=edge_index,
        edge_fraction=t,
        nearest=nearest,
        distance_to_tree=distance,
        mapped=mapped,
        distance_along_edge=distance_along,
        max_distance=max_distance,
        method=method,
    )
    syn._mapping_meta["version"] = MAPPING_VERSION
    if drop_unmapped:
        syn = _drop_unmapped_synapses(syn)
    _bind_synapses_gp(tree, syn)
    return syn


def synapse_mapping_summary(tree: _Tree) -> dict[str, Any]:
    """Return mapping QC counts / distance stats."""
    syn = _require_synapses(tree)
    return syn.summary()


def _node_path_distances(tree: _Tree) -> np.ndarray:
    """Cable path distance from root to every node."""
    lengths = get_edge_length(tree, bind=False)
    g = tree.graph
    # Temporary edge weight property.
    weight = g.new_edge_property("double")
    weight.a = np.asarray(lengths, dtype=np.float64).copy()
    root = int(get_root_index(tree))
    dist = shortest_distance(g, source=g.vertex(root), weights=weight)
    # Copy: dist.a is a view into a temporary PropertyMap.
    return np.asarray(dist.a, dtype=np.float64).copy()


def get_synapse_path_distance(
    tree: _Tree,
    *,
    type: SynapseTypeArg = "both",  # noqa: A002
    mapped_only: bool = True,
    bind: bool = False,
) -> np.ndarray:
    """Path (cable) distance from root to each synapse's continuous edge location.

    For edge ``source → target`` with fraction ``t``::

        path = dist(root, source) + t * edge_length
    """
    syn = _require_synapses(tree)
    if not syn.is_mapped:
        raise ValueError("Synapses must be mapped first (tree.map_synapses())")

    df = syn.to_dataframe(copy=False)
    type_set = resolve_synapse_type_filter(type)
    mask = np.ones(len(df), dtype=bool)
    if type_set is not None:
        mask &= df["type"].isin(type_set).to_numpy()
    if mapped_only:
        mask &= df["mapped"].to_numpy(dtype=bool)

    node_dist = _node_path_distances(tree)
    lengths = np.asarray(get_edge_length(tree, bind=False), dtype=np.float64)
    edges = get_edge_indices(tree)

    out = np.full(len(df), np.nan, dtype=np.float64)
    idx = np.flatnonzero(mask)
    if idx.size:
        eidx = df["edge_index"].to_numpy(dtype=np.int64)[idx]
        sources = edges[eidx, 0]
        if "distance_along_edge" in df.columns:
            along = df["distance_along_edge"].to_numpy(dtype=np.float64)[idx]
            use = np.isfinite(along)
            frac = df["edge_fraction"].to_numpy(dtype=np.float64)[idx]
            local = np.where(use, along, frac * lengths[eidx])
        else:
            frac = df["edge_fraction"].to_numpy(dtype=np.float64)[idx]
            local = frac * lengths[eidx]
        out[idx] = node_dist[sources] + local

    if bind:
        syn.add_column("path_distance", out, overwrite=True)
        _bind_synapses_gp(tree, syn)
    return out


def get_synapse_euclidean_distance_from_root(
    tree: _Tree,
    *,
    type: SynapseTypeArg = "both",  # noqa: A002
    position: Literal["raw", "mapped"] = "raw",
) -> np.ndarray:
    """Euclidean distance from the root coordinate to each synapse."""
    syn = _require_synapses(tree)
    root = get_root_coordinate(tree, SoA=False).ravel()
    if position == "raw":
        coords = syn.coordinates
    elif position == "mapped":
        if not syn.is_mapped:
            raise ValueError("mapped position requires map_synapses()")
        coords = syn.mapped_coordinates
    else:
        raise ValueError("position must be 'raw' or 'mapped'")

    type_set = resolve_synapse_type_filter(type)
    d = np.linalg.norm(coords - root, axis=1)
    if type_set is not None:
        mask = syn.to_dataframe(copy=False)["type"].isin(type_set).to_numpy()
        out = np.full(len(syn), np.nan, dtype=np.float64)
        out[mask] = d[mask]
        return out
    return d


def count_synapses(
    tree: _Tree,
    type: SynapseTypeArg = "both",  # noqa: A002
    *,
    mapped_only: bool = False,
    max_distance: float | None = None,
) -> int:
    """Count attached synapses, optionally filtered."""
    syn = get_synapses(tree)
    if syn is None:
        return 0
    filtered = syn.filter(
        type=type,
        mapped=True if mapped_only else None,
        max_distance=max_distance,
    )
    return len(filtered)


def synapse_density(
    tree: _Tree,
    type: SynapseTypeArg = "both",  # noqa: A002
    *,
    mapped_only: bool = False,
    length_type: Literal["Path", "Euclidean"] = "Path",
) -> float:
    """Synapses per unit cable length (same units as tree coordinates)."""
    n = count_synapses(tree, type=type, mapped_only=mapped_only)
    cable = get_total_cable_length(tree, length_type=length_type)
    if cable <= 0.0:
        raise ValueError("Cannot compute synapse density on zero cable length")
    return float(n) / float(cable)


def get_edge_synapse_counts(
    tree: _Tree,
    type: SynapseTypeArg = "both",  # noqa: A002
    *,
    mapped_only: bool = True,
) -> np.ndarray:
    """Per-edge synapse counts (length = number of edges)."""
    n_edges = tree.graph.num_edges()
    counts = np.zeros(n_edges, dtype=np.int64)
    syn = get_synapses(tree)
    if syn is None or not syn.is_mapped:
        return counts

    filtered = syn.filter(type=type, mapped=True if mapped_only else None)
    if len(filtered) == 0:
        return counts
    eidx = filtered.to_dataframe(copy=False)["edge_index"].to_numpy(dtype=np.int64)
    # Ignore invalid indices if mapping was invalidated inconsistently.
    valid = (eidx >= 0) & (eidx < n_edges)
    eidx = eidx[valid]
    if eidx.size:
        counts += np.bincount(eidx, minlength=n_edges)
    return counts


def get_edge_synapse_density(
    tree: _Tree,
    type: SynapseTypeArg = "both",  # noqa: A002
    *,
    mapped_only: bool = True,
    min_length: float = 1e-12,
) -> np.ndarray:
    """Per-edge synapse density (count / edge length); short edges → 0."""
    counts = get_edge_synapse_counts(tree, type=type, mapped_only=mapped_only)
    lengths = np.asarray(get_edge_length(tree, bind=False), dtype=np.float64)
    dens = np.zeros_like(lengths, dtype=np.float64)
    ok = lengths > float(min_length)
    dens[ok] = counts[ok] / lengths[ok]
    return dens


def filter_tree_synapses_to_edges(
    tree: _Tree,
    edge_indices: np.ndarray,
    *,
    mapped_only: bool = True,
) -> None:
    """Keep synapses whose ``edge_index`` is in *edge_indices*; remap local IDs.

    Used after subtree extraction.

    Parameters
    ----------
    mapped_only
        If True (default), also require ``mapped=True``. If False, keep any row
        whose nearest edge survived, including ``mapped=False`` QC rejects.
    """
    syn = get_synapses(tree)
    if syn is None:
        return
    if not syn.is_mapped:
        # Cannot assign unmapped synapses to retained edges.
        clear_synapses(tree)
        return

    edge_indices = np.asarray(edge_indices, dtype=np.int64)
    old_to_new = {int(old): i for i, old in enumerate(edge_indices)}
    df = syn.to_dataframe(copy=True)
    keep = df["edge_index"].map(lambda e: int(e) in old_to_new).to_numpy(dtype=bool).copy()
    if mapped_only and "mapped" in df.columns:
        keep &= df["mapped"].to_numpy(dtype=bool)
    df = df.loc[keep].reset_index(drop=True)
    if len(df) == 0:
        clear_synapses(tree)
        return
    df["edge_index"] = [old_to_new[int(e)] for e in df["edge_index"]]
    new_syn = Synapses(df, copy=False, mapping_meta=dict(syn.mapping_meta))
    new_syn._mapping_meta["valid"] = True
    _bind_synapses_gp(tree, new_syn)


# synapses_from_arrays is imported for re-export via tree_graphs.__init__.

enrich_tree_graph_docstrings(globals())
