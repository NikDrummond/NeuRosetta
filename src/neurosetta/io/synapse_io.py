"""Tabular synapse import / export.

Two complementary entry points:

* :func:`import_synapses` — load an already per-neuron NeuRosetta-schema table
  (``synapse_id, type, x, y, z, partner_id``).
* :func:`extract_synapses` — reshape a connectivity table (one row per synapse
  with separate pre/post partner IDs and coordinates, FlyWire/navis-style)
  into one :class:`~neurosetta.core.synapses.Synapses` per queried neuron.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np
import pandas as pd

from ..core import _Tree
from ..core.synapses import MAPPING_COLUMNS, Synapses
from ..ops.tree_graphs.tree_synapses import set_synapses
from ..ops.units.synapse_units import apply_synapse_import_units

# Explicit list-likes only — ``str`` / ``bytes`` are Sequence but must stay
# single-id values here.
_LISTLIKE = (list, tuple, np.ndarray, pd.Series, pd.Index)

# Below this many ids, one boolean scan per id beats sorting the whole column.
_SORT_THRESHOLD = 3


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
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
    **read_csv_kwargs: Any,
) -> Synapses:
    """Load synapses from a table and optionally attach them to *tree*.

    Minimum schema (column names remappable via keyword args)::

        synapse_id, type, x, y, z, partner_id

    ``type`` values ``pre``/``post`` (aliases ``output``/``input``). Extra
    columns are preserved as metadata.

    For FlyWire / navis-style connectivity tables (one row per synapse with
    separate pre/post coordinates), use :func:`extract_synapses` first.

    Parameters
    ----------
    set_units : str or None, optional
        Declare spatial units of the loaded coordinates (no rescale), e.g.
        ``\"nm\"``, ``\"micron\"``, or ``\"voxel\"``. Default None.
    voxel_size, voxel_unit
        Required together when ``set_units=\"voxel\"``.
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

    apply_synapse_import_units(syn, set_units, voxel_size=voxel_size, voxel_unit=voxel_unit)

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


def _row_indices(arr: np.ndarray, ids: list, use_sort: bool) -> list[np.ndarray]:
    """For each id, positional row indices where ``arr == id``, original order."""
    if not use_sort:
        return [np.flatnonzero(arr == i) for i in ids]
    q = np.asarray(ids, dtype=arr.dtype)
    # One O(N) pass keeps only rows belonging to any requested id; the sort
    # below then runs on that (usually small) subset, not the whole table.
    cand = np.flatnonzero(np.isin(arr, q))
    sub = arr[cand]
    order = np.argsort(sub, kind="stable")  # stable -> original row order per id
    sorted_sub = sub[order]
    lo = np.searchsorted(sorted_sub, q, side="left")
    hi = np.searchsorted(sorted_sub, q, side="right")
    return [cand[order[a:b]] for a, b in zip(lo, hi, strict=True)]


@overload
def extract_synapses(
    syn_df: pd.DataFrame,
    neuron_id: Hashable,
    *,
    pre_col: str = ...,
    post_col: str = ...,
    syn_id_col: str = ...,
    pre_xyz_cols: Sequence[str] = ...,
    post_xyz_cols: Sequence[str] = ...,
    include_extra: bool = ...,
    set_units: str | None = ...,
    voxel_size: float | None = ...,
    voxel_unit: str | None = ...,
) -> Synapses: ...


@overload
def extract_synapses(
    syn_df: pd.DataFrame,
    neuron_id: list | tuple | np.ndarray | pd.Series | pd.Index,
    *,
    pre_col: str = ...,
    post_col: str = ...,
    syn_id_col: str = ...,
    pre_xyz_cols: Sequence[str] = ...,
    post_xyz_cols: Sequence[str] = ...,
    include_extra: bool = ...,
    set_units: str | None = ...,
    voxel_size: float | None = ...,
    voxel_unit: str | None = ...,
) -> list[Synapses]: ...


def extract_synapses(
    syn_df: pd.DataFrame,
    neuron_id: Hashable | list | tuple | np.ndarray | pd.Series | pd.Index,
    *,
    pre_col: str = "pre",
    post_col: str = "post",
    syn_id_col: str = "id",
    pre_xyz_cols: Sequence[str] = ("pre_x", "pre_y", "pre_z"),
    post_xyz_cols: Sequence[str] = ("post_x", "post_y", "post_z"),
    include_extra: bool = False,
    set_units: str | None = None,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> Synapses | list[Synapses]:
    """Build :class:`~neurosetta.core.synapses.Synapses` from a connectivity table.

    Reshapes a FlyWire / navis-style synapse table — **one row per synapse**
    with separate pre- and postsynaptic partner IDs and coordinates — into the
    NeuRosetta per-neuron schema (one row per synapse *end* belonging to the
    queried neuron).

    Input schema (column names remappable via keyword args)::

        id, pre, post, pre_x, pre_y, pre_z, post_x, post_y, post_z

    Output schema per neuron::

        synapse_id, type, x, y, z, partner_id  (+ extras if include_extra)

    For each queried neuron:

    * rows where the neuron is **post** become ``type="post"`` at the post
      coordinates, with ``partner_id`` = the pre partner;
    * rows where the neuron is **pre** become ``type="pre"`` at the pre
      coordinates, with ``partner_id`` = the post partner;
    * post rows are stacked before pre rows;
    * ``owner_id`` on the resulting :class:`Synapses` is set to that neuron id.

    Use this when you have a connectome edge table. If you already have a
    per-neuron NeuRosetta-schema frame, use :func:`import_synapses` instead.
    To attach results to a whole Forest::

        syns = extract_synapses(connectivity_df, forest.ids(), set_units="nm")
        forest.set_synapses(syns)

    Parameters
    ----------
    syn_df : pandas.DataFrame
        Connectivity table, one row per synapse.
    neuron_id : hashable or list-like of hashable
        Neuron ID(s) to extract. A scalar returns a single ``Synapses``;
        a list / tuple / ndarray / Series / Index returns a ``list`` of
        ``Synapses`` in the same order. Strings stay scalar (not iterated).
    pre_col, post_col : str, optional
        Column names for presynaptic / postsynaptic partner IDs.
        Defaults ``"pre"`` / ``"post"``.
    syn_id_col : str, optional
        Column name for the synapse identifier. Default ``"id"``.
    pre_xyz_cols, post_xyz_cols : sequence of str, optional
        Length-3 column name triples for pre- and postsynaptic coordinates.
        Defaults ``("pre_x", "pre_y", "pre_z")`` and
        ``("post_x", "post_y", "post_z")``.
    include_extra : bool, optional
        If True, append every column of ``syn_df`` that is not one of the
        remapped identity / coordinate columns to each output table.
        Default False.
    set_units : str or None, optional
        Declare spatial units of the extracted coordinates (no rescale), e.g.
        ``\"nm\"``, ``\"micron\"``, or ``\"voxel\"``. Default None.
    voxel_size, voxel_unit
        Required together when ``set_units=\"voxel\"``.

    Returns
    -------
    Synapses or list of Synapses
        Single ``Synapses`` when ``neuron_id`` is scalar; otherwise a list,
        one entry per requested id (empty tables included when a neuron has
        no matching synapses).

    Raises
    ------
    ValueError
        If ``pre_xyz_cols`` or ``post_xyz_cols`` do not each contain exactly
        three names.
    KeyError
        If any required column is missing from ``syn_df``.

    Examples
    --------
    Extract synapses for one neuron::

        syn = extract_synapses(
            connectivity_df,
            neuron_id=720575940621356968,
            set_units="nm",
        )
        tree = nr.set_synapses(tree, syn)

    Batch extract, preserving extra metadata columns::

        tables = extract_synapses(
            connectivity_df,
            neuron_id=[id_a, id_b],
            include_extra=True,
            set_units="nm",
        )

    See Also
    --------
    import_synapses : Load an already per-neuron NeuRosetta-schema table.
    export_synapses : Write a ``Synapses`` table to CSV / Parquet.
    neurosetta.ops.tree_graphs.set_synapses : Attach synapses to a tree.
    neurosetta.ops.forest_ops.forest_synapses.set_synapses :
        Batch-attach to a Forest by ``owner_id`` / mapping key.
    """
    pre_xyz_cols = list(pre_xyz_cols)
    post_xyz_cols = list(post_xyz_cols)
    if len(pre_xyz_cols) != 3 or len(post_xyz_cols) != 3:
        raise ValueError("pre_xyz_cols and post_xyz_cols must each have 3 names.")

    used = [pre_col, post_col, syn_id_col, *pre_xyz_cols, *post_xyz_cols]
    missing = [c for c in used if c not in syn_df.columns]
    if missing:
        raise KeyError(f"Columns missing from syn_df: {missing}")
    extras = [c for c in syn_df.columns if c not in used] if include_extra else []

    single = not isinstance(neuron_id, _LISTLIKE)
    ids = [neuron_id] if single else list(neuron_id)

    # Zero-copy views for single-dtype columns.
    pre_ids = syn_df[pre_col].to_numpy()
    post_ids = syn_df[post_col].to_numpy()
    syn_ids = syn_df[syn_id_col].to_numpy()
    px, py, pz = (syn_df[c].to_numpy() for c in pre_xyz_cols)
    qx, qy, qz = (syn_df[c].to_numpy() for c in post_xyz_cols)

    use_sort = len(ids) >= _SORT_THRESHOLD and pre_ids.dtype != object and post_ids.dtype != object
    post_rows = _row_indices(post_ids, ids, use_sort)
    pre_rows = _row_indices(pre_ids, ids, use_sort)

    types = np.array(["post", "pre"], dtype=object)
    results: list[Synapses] = []
    for nid, post_idx, pre_idx in zip(ids, post_rows, pre_rows, strict=True):
        n_post, n_pre = len(post_idx), len(pre_idx)
        df = pd.DataFrame(
            {
                "synapse_id": np.concatenate([syn_ids[post_idx], syn_ids[pre_idx]]),
                "type": np.repeat(types, [n_post, n_pre]),
                "x": np.concatenate([qx[post_idx], px[pre_idx]]),
                "y": np.concatenate([qy[post_idx], py[pre_idx]]),
                "z": np.concatenate([qz[post_idx], pz[pre_idx]]),
                "partner_id": np.concatenate([pre_ids[post_idx], post_ids[pre_idx]]),
            }
        )
        if extras:
            both = np.concatenate([post_idx, pre_idx])
            df = pd.concat([df, syn_df[extras].take(both).reset_index(drop=True)], axis=1)

        syn = Synapses(df, owner_id=nid)
        apply_synapse_import_units(syn, set_units, voxel_size=voxel_size, voxel_unit=voxel_unit)
        results.append(syn)

    return results[0] if single else results
