"""Tabular synapse import / export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..core import _Tree
from ..core.synapses import MAPPING_COLUMNS, Synapses
from ..ops.tree_graphs.tree_synapses import set_synapses


def _rename_synapse_frame(
    df: pd.DataFrame,
    *,
    coordinate_columns: tuple[str, str, str],
    type_column: str,
    partner_column: str,
    id_column: str,
) -> pd.DataFrame:
    out = df.copy()
    rename: dict[str, str] = {}
    cx, cy, cz = coordinate_columns
    if (cx, cy, cz) != ("x", "y", "z"):
        rename[cx], rename[cy], rename[cz] = "x", "y", "z"
    if type_column != "type":
        rename[type_column] = "type"
    if partner_column != "partner_id" and partner_column in out.columns:
        rename[partner_column] = "partner_id"
    if id_column != "synapse_id" and id_column in out.columns:
        rename[id_column] = "synapse_id"
    if rename:
        out = out.rename(columns=rename)
    if "synapse_id" not in out.columns:
        out.insert(0, "synapse_id", np.arange(len(out)))
    if "partner_id" not in out.columns:
        out["partner_id"] = None
    return out


def import_synapses(
    source: str | Path | pd.DataFrame | Synapses,
    tree: _Tree | None = None,
    *,
    coordinate_columns: tuple[str, str, str] = ("x", "y", "z"),
    type_column: str = "type",
    partner_column: str = "partner_id",
    id_column: str = "synapse_id",
    **read_csv_kwargs: Any,
) -> Synapses:
    """Load synapses from a table and optionally attach them to *tree*.

    Minimum schema (column names remappable via keyword args)::

        synapse_id, type, x, y, z, partner_id

    ``type`` values ``pre``/``post`` (aliases ``output``/``input``). Extra
    columns are preserved as metadata.
    """
    if isinstance(source, Synapses):
        syn = source.copy()
    elif isinstance(source, pd.DataFrame):
        syn = Synapses(
            _rename_synapse_frame(
                source,
                coordinate_columns=coordinate_columns,
                type_column=type_column,
                partner_column=partner_column,
                id_column=id_column,
            )
        )
    else:
        path = Path(source)
        if path.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, **read_csv_kwargs)
        syn = Synapses(
            _rename_synapse_frame(
                df,
                coordinate_columns=coordinate_columns,
                type_column=type_column,
                partner_column=partner_column,
                id_column=id_column,
            )
        )

    if tree is not None:
        return set_synapses(tree, syn)
    return syn


def export_synapses(
    tree_or_synapses: _Tree | Synapses,
    path: str | Path | None = None,
    *,
    include_mapping: bool = True,
) -> pd.DataFrame:
    """Export synapses to a DataFrame and optionally write CSV/Parquet."""
    if isinstance(tree_or_synapses, Synapses):
        syn = tree_or_synapses
    else:
        syn = tree_or_synapses.synapses
        if syn is None:
            raise ValueError("Tree has no synapses to export")

    df = syn.to_dataframe(copy=True)
    if not include_mapping:
        drop = [c for c in MAPPING_COLUMNS if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

    if path is not None:
        out = Path(path)
        if out.suffix.lower() in {".parquet", ".pq"}:
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
    return df
