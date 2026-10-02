"""Forest-level synaptic connectivity tables and graphs.

Accepts a morphology :class:`~neurosetta.core.forest._Forest` **or** a
morphology-free mapping / sequence of :class:`~neurosetta.core.synapses.Synapses`
tables keyed by neuron ID (network-only connectomics).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from graph_tool.all import Graph

from ...core import _Forest, _Tree
from ...core.synapses import Synapses, owner_ids_compatible
from ...ops.tree_graphs.tree_synapses import get_synapses

DeduplicateMode = Literal["auto", "id", "none"]
ConnectivityInput = _Forest | Mapping[Hashable, Synapses] | Sequence[Synapses]
_META_VERTEX_KEYS = ("cell_type", "Neuron_type", "subtype", "hemisphere", "type")


def _resolve_dedupe_column(syn: Synapses, mode: DeduplicateMode) -> str | None:
    cols = set(syn.columns)
    if mode == "none":
        return None
    if mode == "id":
        if "cleft_id" in cols or "cleft_id" in syn.annotations:
            return "cleft_id"
        if "synapse_id" in cols:
            return "synapse_id"
        raise ValueError(
            "deduplicate='id' requires a 'cleft_id' or 'synapse_id' column on synapses"
        )
    # auto: only globally meaningful IDs (cleft_id). Per-tree synapse_id is not
    # assumed unique across the Forest — use deduplicate='id' to force it.
    if "cleft_id" in cols or "cleft_id" in syn.annotations:
        return "cleft_id"
    return None


def _iter_directed_records(
    owner_id: Hashable,
    syn: Synapses,
    *,
    dedupe_col: str | None,
) -> list[tuple[Any, Any, Any]]:
    """Return list of (source_id, target_id, dedupe_key) for one neuron."""
    records: list[tuple[Any, Any, Any]] = []
    types = syn.types
    partners = syn.partner_ids
    if dedupe_col is not None:
        if dedupe_col in syn.annotations:
            keys = syn.annotations[dedupe_col]
        elif dedupe_col == "synapse_id":
            keys = syn.synapse_ids
        else:
            keys = syn._column(dedupe_col)
    else:
        keys = np.array([(owner_id, i) for i in range(len(syn))], dtype=object)

    for i in range(len(syn)):
        partner = partners[i]
        if partner is None or (isinstance(partner, float) and np.isnan(partner)):
            continue
        key = keys[i]
        if types[i] == "pre":
            records.append((owner_id, partner, key))
        else:
            records.append((partner, owner_id, key))
    return records


def _apply_synapse_filter(
    syn: Synapses,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None,
) -> Synapses:
    if synapse_filter is None:
        return syn
    if callable(synapse_filter):
        return synapse_filter(syn)
    if isinstance(synapse_filter, Mapping):
        return syn.filter(**synapse_filter)
    raise TypeError("synapse_filter must be a callable or a mapping of filter kwargs")


def _coerce_owner_id(key: Hashable, syn: Synapses) -> Hashable:
    """Resolve the neuron ID for a table in a mapping/sequence input."""
    if syn.owner_id is None:
        return key
    if not owner_ids_compatible(syn.owner_id, key):
        raise ValueError(f"Synapses.owner_id={syn.owner_id!r} does not match mapping key {key!r}")
    return syn.owner_id if syn.owner_id == key else key


def _numeric_id_equal(a: Hashable, b: Hashable) -> bool:
    """True when *a* and *b* are equal after int coercion (not for arbitrary strs)."""
    if a == b:
        return True
    try:
        return int(a) == int(b)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def _id_in_members(nid: Hashable, member_ids: set[Any]) -> bool:
    """Membership with exact match, falling back to int/str-compatible equality."""
    if nid in member_ids:
        return True
    return any(_numeric_id_equal(nid, mid) for mid in member_ids)


def _resolve_connectivity_entries(
    source: ConnectivityInput,
) -> tuple[list[tuple[Hashable, Synapses]], dict[Any, _Tree | None], set[Any]]:
    """Normalise Forest / mapping / sequence into (owner, syn) entries.

    Returns
    -------
    entries
        Non-empty synapse tables with resolved owner IDs.
    id_to_tree
        Tree objects when *source* is a Forest; otherwise empty.
    member_ids
        All neuron IDs in the population (forest members or mapping keys),
        including those with empty/missing synapse tables.
    """
    id_to_tree: dict[Any, _Tree | None] = {}
    member_ids: set[Any] = set()
    entries: list[tuple[Hashable, Synapses]] = []

    if isinstance(source, _Forest):
        member_ids = {t.ID for t in source}
        id_to_tree = {t.ID: t for t in source}
        for tree in source:
            syn = get_synapses(tree)
            if syn is None or len(syn) == 0:
                continue
            entries.append((tree.ID, syn))
        return entries, id_to_tree, member_ids

    if isinstance(source, Mapping):
        for key, syn in source.items():
            if not isinstance(syn, Synapses):
                raise TypeError(
                    f"Mapping values must be Synapses instances; got {type(syn)!r} for key {key!r}"
                )
            owner = _coerce_owner_id(key, syn)
            if len(syn) > 0:
                entries.append((owner, syn))
        member_ids = set(source.keys())
        return entries, {}, member_ids

    if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        seen: list[Any] = []
        for i, syn in enumerate(source):
            if not isinstance(syn, Synapses):
                raise TypeError(
                    f"Sequence items must be Synapses instances; got {type(syn)!r} at index {i}"
                )
            if syn.owner_id is None:
                raise ValueError(
                    f"Synapses at index {i} has owner_id=None; set owner_id "
                    f"before network-only connectivity, or pass a "
                    f"Mapping[owner_id, Synapses]."
                )
            owner = syn.owner_id
            if any(_numeric_id_equal(owner, s) for s in seen):
                raise ValueError(f"Duplicate Synapses.owner_id={owner!r} in sequence")
            seen.append(owner)
            member_ids.add(owner)
            if len(syn) == 0:
                continue
            entries.append((owner, syn))
        return entries, {}, member_ids

    raise TypeError(
        "connectivity source must be a Forest, Mapping[id, Synapses], "
        f"or Sequence[Synapses]; got {type(source)!r}"
    )


def aggregate_connectivity(
    source: ConnectivityInput,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate directed synaptic counts between neuron IDs.

    Parameters
    ----------
    source
        A :class:`~neurosetta.core.forest._Forest`, a ``Mapping[owner_id,
        Synapses]``, or a ``Sequence[Synapses]`` whose tables each have
        ``owner_id`` set. Morphology is not required for mapping/sequence input.

    Direction
    ---------
    - ``pre`` / output on neuron A with partner B → ``A → B``
    - ``post`` / input on neuron A with partner B → ``B → A``

    Deduplication
    -------------
    ``auto``
        Use ``cleft_id`` when present; otherwise do not cross-deduplicate
        (per-neuron ``synapse_id`` is not assumed globally unique).
    ``id``
        Require ``cleft_id`` or ``synapse_id``; each ID counted once.
    ``none``
        Count every synapse record (may double-count if both neurons store the
        same physical synapse).

    Returns
    -------
    DataFrame
        Columns: ``source_id``, ``target_id``, ``synapse_count``.
    """
    entries, _, member_ids = _resolve_connectivity_entries(source)
    seen_keys: set[Any] = set()
    counts: dict[tuple[Any, Any], int] = {}

    dedupe_col: str | None = None
    if deduplicate != "none":
        for _, syn in entries:
            syn_f = _apply_synapse_filter(syn, synapse_filter)
            if len(syn_f) == 0:
                continue
            try:
                dedupe_col = _resolve_dedupe_column(syn_f, deduplicate)
            except ValueError:
                if deduplicate == "id":
                    raise
                dedupe_col = None
            break

    for owner_id, syn in entries:
        syn = _apply_synapse_filter(syn, synapse_filter)
        if len(syn) == 0:
            continue
        local_col = dedupe_col
        if local_col is None and deduplicate == "auto":
            local_col = _resolve_dedupe_column(syn, "auto")
        elif deduplicate == "id":
            local_col = _resolve_dedupe_column(syn, "id")

        for src, tgt, key in _iter_directed_records(owner_id, syn, dedupe_col=local_col):
            if local_col is not None:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            if not include_external and (
                not _id_in_members(src, member_ids) or not _id_in_members(tgt, member_ids)
            ):
                continue
            pair = (src, tgt)
            counts[pair] = counts.get(pair, 0) + 1

    rows = [
        {"source_id": s, "target_id": t, "synapse_count": c}
        for (s, t), c in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
        if c >= int(min_synapses)
    ]
    return pd.DataFrame(rows, columns=["source_id", "target_id", "synapse_count"])


def get_connectivity_table(
    source: ConnectivityInput,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a directed synapse-count table for *source*.

    See :func:`aggregate_connectivity` for direction, deduplication, and
    accepted input types.
    """
    return aggregate_connectivity(
        source,
        include_external=include_external,
        deduplicate=deduplicate,
        min_synapses=min_synapses,
        synapse_filter=synapse_filter,
    )


def get_connectivity_graph(
    source: ConnectivityInput,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
    table: pd.DataFrame | None = None,
) -> Graph:
    """Build a directed graph-tool connectivity graph for *source*.

    Vertices correspond to neurons (``vp["tree_id"]``). Population members have
    ``vp["in_forest"] = True``; external partners (when requested) have
    ``False``. Edges carry ``ep["synapse_count"]``.

    When *source* is a Forest, optional neuron metadata is copied onto vertices.
    Mapping / sequence inputs build the same network graph without morphology.

    This is a **network** graph (neurons as nodes), not a morphology graph.
    """
    _, id_to_tree, member_ids = _resolve_connectivity_entries(source)
    if table is None:
        table = get_connectivity_table(
            source,
            include_external=include_external,
            deduplicate=deduplicate,
            min_synapses=min_synapses,
            synapse_filter=synapse_filter,
        )

    forest_ids = sorted(member_ids, key=lambda x: str(x))
    vertex_ids: list[Any] = list(forest_ids)
    if include_external and len(table):
        extras = set(table["source_id"]).union(table["target_id"]) - set(forest_ids)
        vertex_ids.extend(sorted(extras, key=lambda x: str(x)))

    g = Graph(directed=True)
    g.add_vertex(len(vertex_ids))
    id_to_v = {nid: g.vertex(i) for i, nid in enumerate(vertex_ids)}

    tree_id = g.new_vertex_property("object")
    in_forest = g.new_vertex_property("bool")
    for i, nid in enumerate(vertex_ids):
        v = g.vertex(i)
        tree_id[v] = nid
        in_forest[v] = _id_in_members(nid, member_ids)
    g.vp["tree_id"] = tree_id
    g.vp["in_forest"] = in_forest

    if id_to_tree:
        for key in _META_VERTEX_KEYS:
            vals = []
            any_present = False
            for nid in vertex_ids:
                tree = id_to_tree.get(nid)
                if tree is None:
                    for mid, t in id_to_tree.items():
                        if _numeric_id_equal(nid, mid):
                            tree = t
                            break
                if tree is not None and key in tree.metadata:
                    vals.append(tree.metadata[key])
                    any_present = True
                else:
                    vals.append(None)
            if any_present:
                prop = g.new_vertex_property("object")
                for i, val in enumerate(vals):
                    prop[g.vertex(i)] = val
                g.vp[key] = prop

    synapse_count = g.new_edge_property("int")
    for _, row in table.iterrows():
        s, t = row["source_id"], row["target_id"]
        if s not in id_to_v or t not in id_to_v:
            continue
        e = g.add_edge(id_to_v[s], id_to_v[t])
        synapse_count[e] = int(row["synapse_count"])
    g.ep["synapse_count"] = synapse_count
    return g


def _member_id_dict(source: ConnectivityInput, fill: int = 0) -> dict[Any, int]:
    _, _, member_ids = _resolve_connectivity_entries(source)
    return {mid: fill for mid in member_ids}


def _strengths_from_table(
    source: ConnectivityInput,
    table: pd.DataFrame,
) -> tuple[dict[Any, int], dict[Any, int]]:
    in_s = _member_id_dict(source, 0)
    out_s = _member_id_dict(source, 0)
    for _, row in table.iterrows():
        s, t, c = row["source_id"], row["target_id"], int(row["synapse_count"])
        if s in out_s:
            out_s[s] += c
        if t in in_s:
            in_s[t] += c
    return in_s, out_s


def _degrees_from_table(
    source: ConnectivityInput,
    table: pd.DataFrame,
) -> tuple[dict[Any, int], dict[Any, int]]:
    members = _member_id_dict(source, 0)
    in_partners: dict[Any, set] = {k: set() for k in members}
    out_partners: dict[Any, set] = {k: set() for k in members}
    for _, row in table.iterrows():
        s, t = row["source_id"], row["target_id"]
        if s in out_partners:
            out_partners[s].add(t)
        if t in in_partners:
            in_partners[t].add(s)
    return (
        {k: len(v) for k, v in in_partners.items()},
        {k: len(v) for k, v in out_partners.items()},
    )


def get_in_degree(source: ConnectivityInput, **kwargs) -> dict[Any, int]:
    """Number of distinct input partner neurons per neuron ID."""
    table = get_connectivity_table(source, **kwargs)
    return _degrees_from_table(source, table)[0]


def get_out_degree(source: ConnectivityInput, **kwargs) -> dict[Any, int]:
    """Number of distinct output partner neurons per neuron ID."""
    table = get_connectivity_table(source, **kwargs)
    return _degrees_from_table(source, table)[1]


def get_in_strength(source: ConnectivityInput, **kwargs) -> dict[Any, int]:
    """Total input synapse count per neuron ID."""
    table = get_connectivity_table(source, **kwargs)
    return _strengths_from_table(source, table)[0]


def get_out_strength(source: ConnectivityInput, **kwargs) -> dict[Any, int]:
    """Total output synapse count per neuron ID."""
    table = get_connectivity_table(source, **kwargs)
    return _strengths_from_table(source, table)[1]
