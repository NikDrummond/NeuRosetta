"""Forest-level synaptic connectivity tables and graphs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import numpy as np
import pandas as pd
from graph_tool.all import Graph

from ...core import _Forest, _Tree
from ...core.synapses import Synapses
from ...ops.tree_graphs.tree_synapses import get_synapses

DeduplicateMode = Literal["auto", "id", "none"]
_META_VERTEX_KEYS = ("cell_type", "Neuron_type", "subtype", "hemisphere", "type")


def _resolve_dedupe_column(syn: Synapses, mode: DeduplicateMode) -> str | None:
    cols = set(syn.columns)
    if mode == "none":
        return None
    if mode == "id":
        if "cleft_id" in cols:
            return "cleft_id"
        if "synapse_id" in cols:
            return "synapse_id"
        raise ValueError(
            "deduplicate='id' requires a 'cleft_id' or 'synapse_id' column on synapses"
        )
    # auto: only globally meaningful IDs (cleft_id). Per-tree synapse_id is not
    # assumed unique across the Forest — use deduplicate='id' to force it.
    if "cleft_id" in cols:
        return "cleft_id"
    return None


def _iter_directed_records(
    tree: _Tree,
    syn: Synapses,
    *,
    dedupe_col: str | None,
) -> list[tuple[Any, Any, Any]]:
    """Return list of (source_id, target_id, dedupe_key) for one tree."""
    df = syn.to_dataframe(copy=False)
    tree_id = tree.ID
    records: list[tuple[Any, Any, Any]] = []
    types = df["type"].to_numpy()
    partners = df["partner_id"].to_numpy()
    if dedupe_col is not None:
        keys = df[dedupe_col].to_numpy()
    else:
        keys = np.arange(len(df))
        # Make keys unique per tree so "none" never collapses across trees.
        keys = np.array([(tree_id, int(k)) for k in keys], dtype=object)

    for i in range(len(df)):
        partner = partners[i]
        if partner is None or (isinstance(partner, float) and np.isnan(partner)):
            continue
        key = keys[i]
        if types[i] == "pre":
            records.append((tree_id, partner, key))
        else:
            records.append((partner, tree_id, key))
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


def aggregate_connectivity(
    forest: _Forest,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Aggregate directed synaptic counts between neuron IDs.

    Direction
    ---------
    - ``pre`` / output on tree A with partner B → ``A → B``
    - ``post`` / input on tree A with partner B → ``B → A``

    Deduplication
    -------------
    ``auto``
        Use ``cleft_id`` when present; otherwise do not cross-deduplicate
        (per-tree ``synapse_id`` is not assumed globally unique).
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
    forest_ids = {int(t.ID) for t in forest}
    seen_keys: set[Any] = set()
    counts: dict[tuple[Any, Any], int] = {}

    # Resolve dedupe column from the first non-empty synapse table that has one.
    dedupe_col: str | None = None
    if deduplicate != "none":
        for tree in forest:
            syn = get_synapses(tree)
            if syn is None or len(syn) == 0:
                continue
            syn = _apply_synapse_filter(syn, synapse_filter)
            try:
                dedupe_col = _resolve_dedupe_column(syn, deduplicate)
            except ValueError:
                if deduplicate == "id":
                    raise
                dedupe_col = None
            break

    for tree in forest:
        syn = get_synapses(tree)
        if syn is None or len(syn) == 0:
            continue
        syn = _apply_synapse_filter(syn, synapse_filter)
        if len(syn) == 0:
            continue
        # Per-tree column if auto and first tree lacked ids.
        local_col = dedupe_col
        if local_col is None and deduplicate == "auto":
            local_col = _resolve_dedupe_column(syn, "auto")
        elif deduplicate == "id":
            local_col = _resolve_dedupe_column(syn, "id")

        for src, tgt, key in _iter_directed_records(tree, syn, dedupe_col=local_col):
            if local_col is not None:
                if key in seen_keys:
                    continue
                seen_keys.add(key)
            if not include_external:
                try:
                    src_i, tgt_i = int(src), int(tgt)
                except (TypeError, ValueError):
                    continue
                if src_i not in forest_ids or tgt_i not in forest_ids:
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
    forest: _Forest,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Return a directed synapse-count table for *forest*.

    See :func:`aggregate_connectivity` for direction and deduplication semantics.
    """
    return aggregate_connectivity(
        forest,
        include_external=include_external,
        deduplicate=deduplicate,
        min_synapses=min_synapses,
        synapse_filter=synapse_filter,
    )


def get_connectivity_graph(
    forest: _Forest,
    *,
    include_external: bool = False,
    deduplicate: DeduplicateMode = "auto",
    min_synapses: int = 1,
    synapse_filter: Callable[[Synapses], Synapses] | Mapping[str, Any] | None = None,
    table: pd.DataFrame | None = None,
) -> Graph:
    """Build a directed graph-tool connectivity graph for *forest*.

    Vertices correspond to neurons (``vp["tree_id"]``). Forest members have
    ``vp["in_forest"] = True``; external partners (when requested) have
    ``False``. Edges carry ``ep["synapse_count"]``.

    This is a **network** graph (neurons as nodes), not a morphology graph.
    """
    if table is None:
        table = get_connectivity_table(
            forest,
            include_external=include_external,
            deduplicate=deduplicate,
            min_synapses=min_synapses,
            synapse_filter=synapse_filter,
        )

    forest_ids = [int(t.ID) for t in forest]
    id_to_tree = {int(t.ID): t for t in forest}

    vertex_ids: list[Any] = list(forest_ids)
    if include_external and len(table):
        extras = set(table["source_id"]).union(table["target_id"]) - set(forest_ids)
        # Stable order: forest first, then sorted external.
        vertex_ids.extend(sorted(extras, key=lambda x: str(x)))

    g = Graph(directed=True)
    g.add_vertex(len(vertex_ids))
    id_to_v = {nid: g.vertex(i) for i, nid in enumerate(vertex_ids)}

    tree_id = g.new_vertex_property("object")
    in_forest = g.new_vertex_property("bool")
    for i, nid in enumerate(vertex_ids):
        v = g.vertex(i)
        tree_id[v] = nid
        in_forest[v] = nid in id_to_tree
    g.vp["tree_id"] = tree_id
    g.vp["in_forest"] = in_forest

    # Optional metadata from Trees.
    for key in _META_VERTEX_KEYS:
        vals = []
        any_present = False
        for nid in vertex_ids:
            tree = id_to_tree.get(int(nid) if not isinstance(nid, int) else nid)
            if tree is None:
                try:
                    tree = id_to_tree.get(int(nid))
                except (TypeError, ValueError):
                    tree = None
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


def _strengths_from_table(
    forest: _Forest,
    table: pd.DataFrame,
) -> tuple[dict[Any, int], dict[Any, int]]:
    in_s: dict[Any, int] = {int(t.ID): 0 for t in forest}
    out_s: dict[Any, int] = {int(t.ID): 0 for t in forest}
    for _, row in table.iterrows():
        s, t, c = row["source_id"], row["target_id"], int(row["synapse_count"])
        if s in out_s:
            out_s[s] += c
        if t in in_s:
            in_s[t] += c
    return in_s, out_s


def _degrees_from_table(
    forest: _Forest,
    table: pd.DataFrame,
) -> tuple[dict[Any, int], dict[Any, int]]:
    in_partners: dict[Any, set] = {int(t.ID): set() for t in forest}
    out_partners: dict[Any, set] = {int(t.ID): set() for t in forest}
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


def get_in_degree(forest: _Forest, **kwargs) -> dict[Any, int]:
    """Number of distinct input partner neurons per Tree ID."""
    table = get_connectivity_table(forest, **kwargs)
    return _degrees_from_table(forest, table)[0]


def get_out_degree(forest: _Forest, **kwargs) -> dict[Any, int]:
    """Number of distinct output partner neurons per Tree ID."""
    table = get_connectivity_table(forest, **kwargs)
    return _degrees_from_table(forest, table)[1]


def get_in_strength(forest: _Forest, **kwargs) -> dict[Any, int]:
    """Total input synapse count per Tree ID."""
    table = get_connectivity_table(forest, **kwargs)
    return _strengths_from_table(forest, table)[0]


def get_out_strength(forest: _Forest, **kwargs) -> dict[Any, int]:
    """Total output synapse count per Tree ID."""
    table = get_connectivity_table(forest, **kwargs)
    return _strengths_from_table(forest, table)[1]
