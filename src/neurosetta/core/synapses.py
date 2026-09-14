"""First-class synapse table attached to neuron morphologies.

Synapses are observations associated with a :class:`~neurosetta.core.tree._Tree`
and are **not** vertices in the morphology graph. The canonical continuous
location of a mapped synapse is ``(edge_index, edge_fraction)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal, Self

import numpy as np
import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "synapse_id",
    "type",
    "x",
    "y",
    "z",
    "partner_id",
)

MAPPING_COLUMNS: tuple[str, ...] = (
    "edge_index",
    "edge_fraction",
    "distance_along_edge",
    "nearest_x",
    "nearest_y",
    "nearest_z",
    "distance_to_tree",
    "mapped",
)

TYPE_ALIASES: dict[str, str] = {
    "pre": "pre",
    "post": "post",
    "output": "pre",
    "outputs": "pre",
    "input": "post",
    "inputs": "post",
}

SynapseType = Literal["pre", "post", "both"]
MAPPING_VERSION = 1


def canonicalize_synapse_type(value: Any) -> str:
    """Map user-facing type labels to ``\"pre\"`` / ``\"post\"``."""
    key = str(value).strip().lower()
    if key not in TYPE_ALIASES:
        raise ValueError(
            f"Invalid synapse type {value!r}; expected one of "
            f"{sorted(set(TYPE_ALIASES))} (aliases for pre/post)."
        )
    return TYPE_ALIASES[key]


def resolve_synapse_type_filter(type_: SynapseType | str | None) -> set[str] | None:
    """Return the set of canonical types selected by a filter, or None for all."""
    if type_ is None or type_ == "both":
        return None
    return {canonicalize_synapse_type(type_)}


def _as_1d(values: Any, name: str, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"{name} length {arr.shape[0]} != expected {n}")
    return arr


def _coords_from_arrays(
    coordinates: np.ndarray | None,
    x: Any,
    y: Any,
    z: Any,
    n: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if coordinates is not None:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must have shape (N, 3), got {coords.shape}")
        if n is not None and coords.shape[0] != n:
            raise ValueError(f"coordinates length {coords.shape[0]} != expected {n}")
        return coords[:, 0], coords[:, 1], coords[:, 2]
    if x is None or y is None or z is None:
        raise ValueError("Provide coordinates=(N,3) or x, y, and z arrays")
    xx = _as_1d(x, "x", n).astype(np.float64, copy=False)
    yy = _as_1d(y, "y", len(xx)).astype(np.float64, copy=False)
    zz = _as_1d(z, "z", len(xx)).astype(np.float64, copy=False)
    return xx, yy, zz


def synapses_from_arrays(
    *,
    synapse_id: Any = None,
    type: Any,  # noqa: A002 — matches public column name
    coordinates: np.ndarray | None = None,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    partner_id: Any = None,
    **extra_columns: Any,
) -> Synapses:
    """Build a :class:`Synapses` table from NumPy-like arrays."""
    types = np.asarray(type, dtype=object)
    if types.ndim != 1:
        raise ValueError(f"type must be 1-D, got shape {types.shape}")
    n = len(types)
    xx, yy, zz = _coords_from_arrays(coordinates, x, y, z, n)

    ids = np.arange(n) if synapse_id is None else _as_1d(synapse_id, "synapse_id", n)

    if partner_id is None:
        partners = np.array([None] * n, dtype=object)
    else:
        partners = _as_1d(partner_id, "partner_id", n)

    data: dict[str, Any] = {
        "synapse_id": ids,
        "type": [canonicalize_synapse_type(t) for t in types],
        "x": xx,
        "y": yy,
        "z": zz,
        "partner_id": partners,
    }
    for key, values in extra_columns.items():
        if key in data:
            raise ValueError(f"Duplicate column {key!r}")
        data[key] = _as_1d(values, key, n)

    return Synapses(pd.DataFrame(data))


class Synapses:
    """Table-oriented synapse container.

    Parameters
    ----------
    data
        DataFrame, mapping of columns, or another :class:`Synapses`.
    copy
        If True (default when constructing from a DataFrame/mapping), copy the
        underlying table. Views created by :meth:`filter` pass ``copy=False``.
    mapping_meta
        Optional provenance for derived mapping fields.
    """

    __slots__ = ("_df", "_mapping_meta")

    def __init__(
        self,
        data: pd.DataFrame | Mapping[str, Any] | Synapses,
        *,
        copy: bool = True,
        mapping_meta: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(data, Synapses):
            self._df = data._df.copy() if copy else data._df
            self._mapping_meta = dict(data._mapping_meta)
            if mapping_meta is not None:
                self._mapping_meta.update(mapping_meta)
            return

        if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy() if copy else data
        else:
            raise TypeError(
                f"Synapses data must be a DataFrame, mapping, or Synapses; got {type(data)!r}"
            )

        self._df = self._normalize_dataframe(df)
        self._mapping_meta = dict(mapping_meta or {})

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Synapses table missing required columns: {missing}")

        out = df.copy()
        out["type"] = [canonicalize_synapse_type(t) for t in out["type"]]
        for col in ("x", "y", "z"):
            out[col] = pd.to_numeric(out[col], errors="raise").astype(np.float64)

        if out["synapse_id"].duplicated().any():
            dups = out.loc[out["synapse_id"].duplicated(), "synapse_id"].tolist()
            raise ValueError(f"Duplicate synapse_id values: {dups[:10]}")

        if "mapped" in out.columns:
            out["mapped"] = out["mapped"].astype(bool)
        return out.reset_index(drop=True)

    # --- pickle / graph-tool object gp ---

    def __getstate__(self) -> dict[str, Any]:
        return {"df": self._df, "mapping_meta": self._mapping_meta}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._df = state["df"]
        self._mapping_meta = dict(state.get("mapping_meta") or {})

    # --- basics ---

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        n_pre = int((self._df["type"] == "pre").sum())
        n_post = int((self._df["type"] == "post").sum())
        mapped = "mapped" if self.is_mapped else "unmapped"
        return f"Synapses(n={len(self)}, pre={n_pre}, post={n_post}, {mapped})"

    def copy(self) -> Self:
        """Deep-copy the table and mapping metadata."""
        return self.__class__(self._df, copy=True, mapping_meta=dict(self._mapping_meta))

    def to_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the underlying table."""
        return self._df.copy() if copy else self._df

    @property
    def columns(self) -> list[str]:
        return list(self._df.columns)

    @property
    def mapping_meta(self) -> dict[str, Any]:
        return self._mapping_meta

    @property
    def is_mapped(self) -> bool:
        """True when a full mapping column set is present."""
        return all(c in self._df.columns for c in MAPPING_COLUMNS)

    @property
    def mapping_valid(self) -> bool:
        """True when mapping columns exist and are marked valid."""
        return bool(self.is_mapped and self._mapping_meta.get("valid", False))

    # --- coordinates ---

    @property
    def coordinates(self) -> np.ndarray:
        """Raw synapse coordinates as ``(N, 3)`` float64."""
        return self._df.loc[:, ["x", "y", "z"]].to_numpy(dtype=np.float64, copy=True)

    @property
    def mapped_coordinates(self) -> np.ndarray:
        """Nearest points on the morphology; NaN where unmapped."""
        if not self.is_mapped:
            raise ValueError("Synapses are not mapped; call tree.map_synapses() first")
        return self._df.loc[:, ["nearest_x", "nearest_y", "nearest_z"]].to_numpy(
            dtype=np.float64, copy=True
        )

    # --- type views ---

    @property
    def pre(self) -> Self:
        """Presynaptic / output synapses."""
        return self.filter(type="pre")

    @property
    def post(self) -> Self:
        """Postsynaptic / input synapses."""
        return self.filter(type="post")

    @property
    def outputs(self) -> Self:
        """Alias for :attr:`pre`."""
        return self.pre

    @property
    def inputs(self) -> Self:
        """Alias for :attr:`post`."""
        return self.post

    # --- filtering ---

    def filter(
        self,
        *,
        type: SynapseType | str | None = None,  # noqa: A002
        partner_id: Any = None,
        mapped: bool | None = None,
        max_distance: float | None = None,
        synapse_id: Any = None,
        **column_equals: Any,
    ) -> Self:
        """Return a filtered copy (does not mutate this collection).

        Parameters
        ----------
        type
            ``\"pre\"``, ``\"post\"``, ``\"both\"`` / ``None``, or aliases
            ``input`` / ``output``.
        partner_id
            Scalar partner or iterable of partners.
        mapped
            If set, require ``mapped`` column equal to this value.
        max_distance
            Keep synapses with ``distance_to_tree <= max_distance`` (mapped only).
        synapse_id
            Scalar ID or iterable of IDs.
        **column_equals
            Additional ``column=value`` equality filters (iterables allowed).
        """
        mask = np.ones(len(self._df), dtype=bool)

        type_set = resolve_synapse_type_filter(type)
        if type_set is not None:
            mask &= self._df["type"].isin(type_set).to_numpy()

        if partner_id is not None:
            mask &= self._isin_mask("partner_id", partner_id)

        if synapse_id is not None:
            mask &= self._isin_mask("synapse_id", synapse_id)

        if mapped is not None:
            if "mapped" not in self._df.columns:
                if mapped:
                    mask[:] = False
            else:
                mask &= self._df["mapped"].to_numpy(dtype=bool) == bool(mapped)

        if max_distance is not None:
            if "distance_to_tree" not in self._df.columns:
                raise ValueError("max_distance requires mapped synapses")
            dist = self._df["distance_to_tree"].to_numpy(dtype=np.float64)
            mask &= np.isfinite(dist) & (dist <= float(max_distance))

        for col, value in column_equals.items():
            if col not in self._df.columns:
                raise KeyError(f"Unknown synapse column {col!r}")
            mask &= self._isin_mask(col, value)

        return self.__class__(
            self._df.loc[mask].reset_index(drop=True),
            copy=False,
            mapping_meta=dict(self._mapping_meta),
        )

    def _isin_mask(self, column: str, value: Any) -> np.ndarray:
        series = self._df[column]
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            return (series == value).to_numpy()
        values = list(value)
        return series.isin(values).to_numpy()

    # --- mutation helpers (used by ops) ---

    def clear_mapping(self) -> None:
        """Drop derived mapping columns and mark mapping invalid."""
        drop = [c for c in MAPPING_COLUMNS if c in self._df.columns]
        if drop:
            self._df.drop(columns=drop, inplace=True)
        self._mapping_meta = {}

    def set_mapping(
        self,
        *,
        edge_index: np.ndarray,
        edge_fraction: np.ndarray,
        nearest: np.ndarray,
        distance_to_tree: np.ndarray,
        mapped: np.ndarray,
        distance_along_edge: np.ndarray | None = None,
        max_distance: float | None = None,
        method: str = "project_points_to_segments",
    ) -> None:
        """Write mapping columns in place."""
        n = len(self._df)
        edge_index = _as_1d(edge_index, "edge_index", n)
        edge_fraction = _as_1d(edge_fraction, "edge_fraction", n).astype(np.float64)
        distance_to_tree = _as_1d(distance_to_tree, "distance_to_tree", n).astype(np.float64)
        mapped = _as_1d(mapped, "mapped", n).astype(bool)
        nearest = np.asarray(nearest, dtype=np.float64)
        if nearest.shape != (n, 3):
            raise ValueError(f"nearest must have shape ({n}, 3), got {nearest.shape}")
        if distance_along_edge is None:
            # Caller may fill later; keep NaN until lengths are known.
            distance_along_edge = np.full(n, np.nan, dtype=np.float64)
        else:
            distance_along_edge = _as_1d(distance_along_edge, "distance_along_edge", n).astype(
                np.float64
            )

        self._df["edge_index"] = edge_index.astype(np.int64)
        self._df["edge_fraction"] = edge_fraction
        self._df["distance_along_edge"] = distance_along_edge
        self._df["nearest_x"] = nearest[:, 0]
        self._df["nearest_y"] = nearest[:, 1]
        self._df["nearest_z"] = nearest[:, 2]
        self._df["distance_to_tree"] = distance_to_tree
        self._df["mapped"] = mapped
        self._mapping_meta = {
            "valid": True,
            "method": method,
            "version": MAPPING_VERSION,
            "max_distance": max_distance,
        }

    def remap_edges_through_reduction(self, reduction_map) -> None:
        """Transfer continuous edge locations through a :class:`ReductionMap`.

        Preserves raw coordinates, mapped coordinates, and ``distance_to_tree``.
        Updates ``edge_index``, ``edge_fraction``, and ``distance_along_edge`` to
        refer to reduced section edges (cable distance along the section).
        """
        if not self.is_mapped:
            return
        df = self._df
        n = len(df)
        new_edges = np.empty(n, dtype=np.int64)
        new_fracs = np.empty(n, dtype=np.float64)
        new_dists = np.empty(n, dtype=np.float64)
        old_edges = df["edge_index"].to_numpy(dtype=np.int64)
        fracs = df["edge_fraction"].to_numpy(dtype=np.float64)
        has_local = "distance_along_edge" in df.columns
        locals_ = (
            df["distance_along_edge"].to_numpy(dtype=np.float64)
            if has_local
            else np.full(n, np.nan)
        )

        for i in range(n):
            eidx = int(old_edges[i])
            try:
                new_e, offset, length, section_length = reduction_map.old_edge_to_section[eidx]
            except KeyError as exc:
                raise KeyError(f"Original edge {eidx} missing from reduction map") from exc

            if np.isfinite(locals_[i]):
                local = float(np.clip(locals_[i], 0.0, length))
            else:
                local = float(np.clip(fracs[i], 0.0, 1.0)) * length
            section_dist = offset + local
            if section_length <= 0.0:
                new_frac = 0.0
            else:
                new_frac = float(np.clip(section_dist / section_length, 0.0, 1.0))
            new_edges[i] = new_e
            new_fracs[i] = new_frac
            new_dists[i] = section_dist

        self._df["edge_index"] = new_edges
        self._df["edge_fraction"] = new_fracs
        self._df["distance_along_edge"] = new_dists
        meta = dict(self._mapping_meta)
        meta["valid"] = True
        meta["method"] = "reduction_transfer"
        self._mapping_meta = meta

    def transform_coordinates(
        self,
        *,
        matrix: np.ndarray | None = None,
        translation: Sequence[float] | None = None,
        scale: float | None = None,
    ) -> None:
        """Apply an affine transform to raw (and mapped) coordinates in place.

        ``matrix`` is a ``(3, 3)`` linear map applied before translation.
        ``scale`` uniformly scales coordinates and ``distance_to_tree``.
        """
        coords = self.coordinates
        if matrix is not None:
            m = np.asarray(matrix, dtype=np.float64)
            if m.shape != (3, 3):
                raise ValueError(f"matrix must be (3, 3), got {m.shape}")
            coords = coords @ m.T
        if scale is not None:
            coords = coords * float(scale)
        if translation is not None:
            t = np.asarray(translation, dtype=np.float64).reshape(3)
            coords = coords + t

        self._df["x"] = coords[:, 0]
        self._df["y"] = coords[:, 1]
        self._df["z"] = coords[:, 2]

        if self.is_mapped:
            nearest = self.mapped_coordinates
            if matrix is not None:
                nearest = nearest @ m.T
            if scale is not None:
                nearest = nearest * float(scale)
                self._df["distance_to_tree"] = self._df["distance_to_tree"].to_numpy(
                    dtype=np.float64
                ) * float(scale)
            if translation is not None:
                nearest = nearest + t
            self._df["nearest_x"] = nearest[:, 0]
            self._df["nearest_y"] = nearest[:, 1]
            self._df["nearest_z"] = nearest[:, 2]

    def add_column(self, name: str, values: Any, *, overwrite: bool = False) -> None:
        """Attach an arbitrary annotation column."""
        if name in REQUIRED_COLUMNS:
            raise ValueError(f"Cannot overwrite required column {name!r}")
        if name in self._df.columns and not overwrite:
            raise ValueError(f"Column {name!r} already exists; pass overwrite=True")
        self._df[name] = _as_1d(values, name, len(self._df))

    def summary(self) -> dict[str, Any]:
        """Compact counts useful for QC."""
        dist = None
        n_mapped = 0
        n_unmapped = len(self)
        if self.is_mapped:
            mapped_mask = self._df["mapped"].to_numpy(dtype=bool)
            n_mapped = int(mapped_mask.sum())
            n_unmapped = int((~mapped_mask).sum())
            d = self._df.loc[mapped_mask, "distance_to_tree"].to_numpy(dtype=np.float64)
            dist = d if d.size else None

        out: dict[str, Any] = {
            "n_synapses": len(self),
            "n_pre": int((self._df["type"] == "pre").sum()),
            "n_post": int((self._df["type"] == "post").sum()),
            "n_mapped": n_mapped,
            "n_unmapped": n_unmapped,
            "median_distance_to_tree": float(np.median(dist)) if dist is not None else None,
            "max_distance_to_tree": float(np.max(dist)) if dist is not None else None,
            "mapping_valid": self.mapping_valid,
        }
        return out
