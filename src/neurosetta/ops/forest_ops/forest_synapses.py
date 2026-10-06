"""Attach synapse tables to Forest members by ID."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal

import pandas as pd

from ...core import _Forest
from ...core.synapses import Synapses, owner_ids_compatible
from ...ops.tree_graphs.tree_synapses import set_synapses as _set_synapses_tree

MissingPolicy = Literal["error", "warn", "ignore"]
UnusedPolicy = Literal["error", "warn", "ignore"]

SynapseSource = Synapses | pd.DataFrame | Mapping[str, Any]


def _ids_match(syn_id: Hashable | None, tree_id: Hashable) -> bool:
    """True when synapse match ID and tree ID refer to the same neuron."""
    # owner_ids_compatible(None, …) is True for unbound tables — not wanted here.
    if syn_id is None:
        return False
    return owner_ids_compatible(syn_id, tree_id)


def _as_synapse_entries(
    synapses: Mapping[Hashable, SynapseSource] | Iterable[Synapses] | Synapses,
) -> list[tuple[Hashable, SynapseSource]]:
    """Normalize input to ``(match_id, payload)`` pairs.

    Mapping keys are the match IDs. Iterable / single ``Synapses`` match on
    ``owner_id`` (required).
    """
    if isinstance(synapses, Synapses):
        if synapses.owner_id is None:
            raise ValueError(
                "Synapses.owner_id is required for Forest.set_synapses matching; "
                "set owner_id or pass a {tree_id: synapses} mapping."
            )
        return [(synapses.owner_id, synapses)]

    if isinstance(synapses, Mapping) and not isinstance(synapses, pd.DataFrame):
        out: list[tuple[Hashable, SynapseSource]] = []
        for key, payload in synapses.items():
            if isinstance(payload, (Synapses, pd.DataFrame, Mapping)):
                out.append((key, payload))
            else:
                raise TypeError(
                    f"Mapping values must be Synapses, DataFrame, or column mapping; "
                    f"got {type(payload)!r} for key {key!r}"
                )
        return out

    if isinstance(synapses, pd.DataFrame):
        raise TypeError(
            "Forest.set_synapses does not accept a raw DataFrame. "
            "Use extract_synapses(df, forest.ids()) for connectivity tables, "
            "or pass a {tree_id: dataframe} mapping for NeuRosetta-schema tables."
        )

    items = list(synapses)
    out = []
    for i, item in enumerate(items):
        if not isinstance(item, Synapses):
            raise TypeError(
                f"Expected Synapses members in iterable; got {type(item)!r} at index {i}"
            )
        if item.owner_id is None:
            raise ValueError(
                f"Synapses at index {i} has owner_id=None; "
                f"set owner_id or pass a {{tree_id: synapses}} mapping."
            )
        out.append((item.owner_id, item))
    return out


def _match_entry(
    tree_id: Hashable, entries: list[tuple[Hashable, SynapseSource]]
) -> tuple[Hashable, SynapseSource] | None:
    hits = [(mid, payload) for mid, payload in entries if _ids_match(mid, tree_id)]
    if not hits:
        return None
    if len(hits) > 1:
        ids = [mid for mid, _ in hits]
        raise ValueError(
            f"Multiple synapse tables match tree ID {tree_id!r}: {ids}. "
            f"Ensure owner_id / mapping keys are unique per neuron."
        )
    return hits[0]


def set_synapses(
    forest: _Forest,
    synapses: Mapping[Hashable, SynapseSource] | Iterable[Synapses] | Synapses,
    *,
    missing: MissingPolicy = "warn",
    unused: UnusedPolicy = "ignore",
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> dict[Hashable, Synapses]:
    """Attach synapse tables to Forest trees by matching logical ``ID``.

    Canonical pipeline from a FlyWire / navis-style connectivity table::

        syns = extract_synapses(connectivity_df, forest.ids(), set_units="nm")
        forest.set_synapses(syns)

    Parameters
    ----------
    forest : Forest
        Target morphology collection.
    synapses : Synapses, mapping, or iterable of Synapses
        Tables to attach.

        * ``Iterable[Synapses]`` / single ``Synapses`` — matched by
          ``owner_id`` (must be set; int/str coercion as for meshes).
        * ``Mapping[ID, Synapses | DataFrame | dict]`` — matched by mapping
          key; DataFrame / dict values must already be NeuRosetta-schema
          (``synapse_id, type, x, y, z, partner_id``).

        A raw connectivity DataFrame is rejected — call
        :func:`~neurosetta.io.synapse_io.extract_synapses` first.
    missing : {\"error\", \"warn\", \"ignore\"}, optional
        Policy when a tree has no matching table. By default ``\"warn\"``.
    unused : {\"error\", \"warn\", \"ignore\"}, optional
        Policy when a table matches no tree. By default ``\"ignore\"``.
    set_units : str or None, optional
        Declare spatial units of incoming coordinates before bind
        (no rescale). Forwarded to each tree-level attach. Default None.
    voxel_size, voxel_unit
        Required together when ``set_units=\"voxel\"``.

    Returns
    -------
    dict
        ``{tree.ID: bound Synapses}`` for successfully attached members.

    See Also
    --------
    neurosetta.io.synapse_io.extract_synapses
        Connectivity table → per-neuron ``Synapses`` list.
    neurosetta.ops.tree_graphs.tree_synapses.set_synapses
        Single-tree attach.
    neurosetta.ops.forest_ops.forest_meshes.set_meshes
        Analogous ID-matched mesh attach.
    """
    if missing not in ("error", "warn", "ignore"):
        raise ValueError(f"missing={missing!r} invalid")
    if unused not in ("error", "warn", "ignore"):
        raise ValueError(f"unused={unused!r} invalid")

    entries = _as_synapse_entries(synapses)
    attached: dict[Hashable, Synapses] = {}
    used: set[int] = set()

    for tree in forest:
        match = _match_entry(tree.ID, entries)
        if match is None:
            msg = f"No synapses found for tree ID {tree.ID!r}"
            if missing == "error":
                raise ValueError(msg)
            if missing == "warn":
                warnings.warn(msg, UserWarning, stacklevel=2)
            continue
        match_id, payload = match
        bound = _set_synapses_tree(
            tree,
            payload,
            set_units=set_units,
            voxel_size=voxel_size,
            voxel_unit=voxel_unit,
        )
        attached[tree.ID] = bound
        used.add(id(payload))

    unused_entries = [(mid, p) for mid, p in entries if id(p) not in used]
    if unused_entries:
        ids = [mid for mid, _ in unused_entries]
        msg = f"Synapse tables with no matching tree were unused: {ids}"
        if unused == "error":
            raise ValueError(msg)
        if unused == "warn":
            warnings.warn(msg, UserWarning, stacklevel=2)

    return attached


__all__ = ["set_synapses"]
