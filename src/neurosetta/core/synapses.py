"""First-class synapse table attached to neuron morphologies.

Synapses are observations associated with a :class:`~neurosetta.core.tree._Tree`
and are **not** vertices in the morphology graph. The canonical continuous
location of a mapped synapse is ``(edge_index, edge_fraction)``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
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

TYPE_PRE: np.uint8 = np.uint8(0)
TYPE_POST: np.uint8 = np.uint8(1)
_TYPE_STR_TO_CODE: dict[str, np.uint8] = {"pre": TYPE_PRE, "post": TYPE_POST}
_TYPE_CODE_TO_STR: dict[int, str] = {int(TYPE_PRE): "pre", int(TYPE_POST): "post"}

SynapseType = Literal["pre", "post", "both"]
MAPPING_VERSION = 1
PAYLOAD_FORMAT = "neurosetta.synapses"
PAYLOAD_VERSION = 2


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


def _type_codes_from_values(values: Any, n: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    if arr.ndim != 1:
        raise ValueError(f"type must be 1-D, got shape {arr.shape}")
    if n is not None and arr.shape[0] != n:
        raise ValueError(f"type length {arr.shape[0]} != expected {n}")
    codes = np.empty(len(arr), dtype=np.uint8)
    for i, v in enumerate(arr):
        if isinstance(v, (int, np.integer)) and int(v) in _TYPE_CODE_TO_STR:
            codes[i] = np.uint8(v)
        else:
            codes[i] = _TYPE_STR_TO_CODE[canonicalize_synapse_type(v)]
    return codes


def _types_from_codes(codes: np.ndarray) -> np.ndarray:
    out = np.empty(len(codes), dtype=object)
    for i, c in enumerate(codes):
        out[i] = _TYPE_CODE_TO_STR[int(c)]
    return out


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
) -> np.ndarray:
    if coordinates is not None:
        coords = np.asarray(coordinates, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"coordinates must have shape (N, 3), got {coords.shape}")
        if n is not None and coords.shape[0] != n:
            raise ValueError(f"coordinates length {coords.shape[0]} != expected {n}")
        return np.ascontiguousarray(coords, dtype=np.float64)
    if x is None or y is None or z is None:
        raise ValueError("Provide coordinates=(N,3) or x, y, and z arrays")
    xx = _as_1d(x, "x", n).astype(np.float64, copy=False)
    yy = _as_1d(y, "y", len(xx)).astype(np.float64, copy=False)
    zz = _as_1d(z, "z", len(xx)).astype(np.float64, copy=False)
    return np.column_stack([xx, yy, zz])


def owner_ids_compatible(owner_id: Hashable | None, tree_id: Hashable) -> bool:
    """Return True when *owner_id* is unset or refers to the same neuron as *tree_id*.

    Exact equality is preferred. Numeric IDs also match across ``int`` /
    ``np.integer`` / digit-strings (common connectomics footgun).
    """
    if owner_id is None:
        return True
    if owner_id == tree_id:
        return True
    try:
        return int(owner_id) == int(tree_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def synapses_from_arrays(
    *,
    synapse_id: Any = None,
    type: Any,  # noqa: A002 - matches public column name
    coordinates: np.ndarray | None = None,
    x: Any = None,
    y: Any = None,
    z: Any = None,
    partner_id: Any = None,
    owner_id: Hashable | None = None,
    units: str | None = None,
    **extra_columns: Any,
) -> Synapses:
    """Build a :class:`Synapses` table from NumPy-like arrays."""
    return Synapses.from_arrays(
        synapse_id=synapse_id,
        type=type,
        coordinates=coordinates,
        x=x,
        y=y,
        z=z,
        partner_id=partner_id,
        owner_id=owner_id,
        units=units,
        **extra_columns,
    )


def _empty_object(n: int) -> np.ndarray:
    return np.array([None] * n, dtype=object)


def _copy_array(arr: np.ndarray | None, *, copy: bool) -> np.ndarray | None:
    if arr is None:
        return None
    return arr.copy() if copy else arr


class Synapses:
    """Table of synaptic observations attached to a morphology.

    Synapses are **not** vertices in the morphology graph. They hang off a
    :class:`~neurosetta.core.tree._Tree` as a facet (``tree.synapses``), same
    pattern as ``tree.mesh``. Raw sites are observed ``(x, y, z)``; after
    :func:`~neurosetta.ops.tree_graphs.tree_synapses.map_synapses` the
    morphology-relative location is the continuous pair
    ``(edge_index, edge_fraction)`` (plus cable ``distance_along_edge``).

    Required columns
    ----------------
    ``synapse_id``, ``type``, ``x``, ``y``, ``z``, ``partner_id``

    ``type`` is canonicalized to ``\"pre\"`` / ``\"post\"`` (aliases:
    ``output`` / ``outputs`` → pre; ``input`` / ``inputs`` → post).
    ``synapse_id`` values must be unique. Extra annotation columns are allowed.

    Mapping columns (written by ``map_synapses``)
    ---------------------------------------------
    ``edge_index``, ``edge_fraction``, ``distance_along_edge``,
    ``nearest_x``, ``nearest_y``, ``nearest_z``, ``distance_to_tree``,
    ``mapped``

    Parameters
    ----------
    data
        DataFrame, column mapping, or another :class:`Synapses`. DataFrames /
        mappings are validated for required columns, unique ``synapse_id``,
        and numeric ``x,y,z``.
    copy
        If True (default when constructing from a DataFrame/mapping), copy the
        underlying arrays. Views from :meth:`filter` / :meth:`take` pass
        ``copy=False``.
    mapping_meta
        Optional provenance for derived mapping fields (method, version,
        ``valid``, ``max_distance``, …).
    owner_id
        Neuron / tree ID this table belongs to. ``None`` means unbound.
        :func:`~neurosetta.ops.tree_graphs.tree_synapses.set_synapses` rejects a
        mismatch with ``tree.ID`` (int/str coerce via
        :func:`owner_ids_compatible`) and stamps ``owner_id = tree.ID`` on bind.
    units
        Spatial units of raw ``x,y,z``. ``None`` means unset (warned on
        bind/map). Stamped from the tree when binding if the tree has declared
        units. :meth:`set_units` declares only — it does **not** rescale;
        rescale via tree ``convert_units`` after bind.
    units_meta
        Extra unit metadata (voxel size/unit when ``units == \"voxel\"``).

    Attributes
    ----------
    owner_id
        Bound neuron ID, or ``None`` if unbound.
    units, units_meta
        Spatial unit declaration and voxel metadata.
    columns
        Current logical column names (required + annotations + mapping).
    annotations
        Extra named annotation arrays (not required / mapping columns).
    mapping_meta
        Mapping provenance dict (empty when unmapped).
    is_mapped
        True when the full mapping column set is present.
    mapping_valid
        True when mapped and ``mapping_meta[\"valid\"]`` is set.
    coordinates
        Raw ``(N, 3)`` float64 array from ``x,y,z``.
    mapped_coordinates
        Nearest points on the morphology (requires mapping).
    pre, post, outputs, inputs
        Type-filtered views (``outputs`` ≡ ``pre``, ``inputs`` ≡ ``post``).

    Notes
    -----
    - **Bind:** ``tree.set_synapses(...)`` / ``tree.add_synapses(...)``.
    - **Map:** ``tree.map_synapses(...)`` fills mapping columns in place.
    - **Reduction:** if already mapped, locations transfer via
      :meth:`remap_edges_through_reduction` (map before reduce).
    - **Network:** connectivity ops consume attached tables (or a
      ``dict[ID → Synapses]``) — never merged into the morphology graph.
    - **Persistence:** pickled into ``.nr`` as a graph-tool object property
      via a versioned SoA payload (``PAYLOAD_FORMAT`` / ``PAYLOAD_VERSION``);
      legacy v1 ``{\"df\", ...}`` states still load. Runtime indexes are
      rebuilt on unpickle and are not serialized.

    See Also
    --------
    synapses_from_arrays
        Build a table from NumPy-like arrays.
    neurosetta.ops.tree_graphs.tree_synapses.set_synapses
    neurosetta.ops.tree_graphs.tree_synapses.map_synapses
    neurosetta.ops.forest_ops.forest_connectivity.get_connectivity_graph

    Examples
    --------
    >>> import pandas as pd
    >>> from neurosetta.core.synapses import Synapses
    >>> df = pd.DataFrame({
    ...     "synapse_id": [1, 2],
    ...     "type": ["pre", "post"],
    ...     "x": [0.0, 1.0], "y": [0.0, 0.0], "z": [0.0, 0.0],
    ...     "partner_id": [10, 20],
    ... })
    >>> syn = Synapses(df, owner_id=7, units="nm")
    >>> syn.pre  # doctest: +ELLIPSIS
    Synapses(n=1, pre=1, post=0, unmapped, owner_id=7, units='...')
    """

    __slots__ = (
        "_synapse_id",
        "_type_code",
        "_xyz",
        "_partner_id",
        "_edge_index",
        "_edge_fraction",
        "_distance_along_edge",
        "_nearest",
        "_distance_to_tree",
        "_mapped",
        "_annotations",
        "_mapping_meta",
        "_owner_id",
        "_units",
        "_units_meta",
        "_id_to_row",
        "_partner_to_rows",
        "_edge_to_rows",
    )

    def __init__(
        self,
        data: pd.DataFrame | Mapping[str, Any] | Synapses | None = None,
        *,
        copy: bool = True,
        mapping_meta: dict[str, Any] | None = None,
        owner_id: Hashable | None = None,
        units: str | None = None,
        units_meta: Mapping[str, Any] | None = None,
        _arrays: dict[str, Any] | None = None,
    ) -> None:
        if _arrays is not None:
            self._init_from_arrays(
                _arrays,
                copy=copy,
                mapping_meta=mapping_meta,
                owner_id=owner_id,
                units=units,
                units_meta=units_meta,
            )
            return

        if data is None:
            raise TypeError("Synapses requires data= or _arrays=")

        if isinstance(data, Synapses):
            self._init_from_arrays(
                data._arrays_dict(copy=False),
                copy=copy,
                mapping_meta=(
                    dict(data._mapping_meta)
                    if mapping_meta is None
                    else {
                        **dict(data._mapping_meta),
                        **mapping_meta,
                    }
                ),
                owner_id=data._owner_id if owner_id is None else owner_id,
                units=data._units if units is None else units,
                units_meta=(
                    dict(data._units_meta) if units_meta is None else dict(units_meta)
                ),
            )
            return

        if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy() if copy else data
        else:
            raise TypeError(
                f"Synapses data must be a DataFrame, mapping, or Synapses; got {type(data)!r}"
            )

        self._init_from_dataframe(
            df,
            mapping_meta=mapping_meta,
            owner_id=owner_id,
            units=units,
            units_meta=units_meta,
        )

    def _init_empty_indexes(self) -> None:
        self._id_to_row: dict[Any, int] = {}
        self._partner_to_rows: dict[Any, np.ndarray] = {}
        self._edge_to_rows: dict[int, np.ndarray] = {}

    def _clear_mapping_arrays(self) -> None:
        self._edge_index = None
        self._edge_fraction = None
        self._distance_along_edge = None
        self._nearest = None
        self._distance_to_tree = None
        self._mapped = None

    def _init_from_arrays(
        self,
        arrays: Mapping[str, Any],
        *,
        copy: bool,
        mapping_meta: dict[str, Any] | None,
        owner_id: Hashable | None,
        units: str | None,
        units_meta: Mapping[str, Any] | None,
    ) -> None:
        self._synapse_id = _copy_array(np.asarray(arrays["synapse_id"]), copy=copy)
        self._type_code = _copy_array(
            np.asarray(arrays["type_code"], dtype=np.uint8), copy=copy
        )
        self._xyz = _copy_array(np.asarray(arrays["xyz"], dtype=np.float64), copy=copy)
        self._partner_id = _copy_array(
            np.asarray(arrays["partner_id"], dtype=object), copy=copy
        )

        n = len(self._synapse_id)
        if self._type_code is None or self._xyz is None or self._partner_id is None:
            raise ValueError("Incomplete _arrays payload")
        if self._xyz.shape != (n, 3):
            raise ValueError(f"xyz must have shape ({n}, 3), got {self._xyz.shape}")
        if len(self._type_code) != n or len(self._partner_id) != n:
            raise ValueError("Array length mismatch in _arrays")

        if arrays.get("edge_index") is not None:
            self._edge_index = _copy_array(
                np.asarray(arrays["edge_index"], dtype=np.int64), copy=copy
            )
            self._edge_fraction = _copy_array(
                np.asarray(arrays["edge_fraction"], dtype=np.float64), copy=copy
            )
            self._distance_along_edge = _copy_array(
                np.asarray(arrays["distance_along_edge"], dtype=np.float64), copy=copy
            )
            self._nearest = _copy_array(
                np.asarray(arrays["nearest"], dtype=np.float64), copy=copy
            )
            self._distance_to_tree = _copy_array(
                np.asarray(arrays["distance_to_tree"], dtype=np.float64), copy=copy
            )
            self._mapped = _copy_array(
                np.asarray(arrays["mapped"], dtype=bool), copy=copy
            )
        else:
            self._clear_mapping_arrays()

        ann_src = arrays.get("annotations") or {}
        self._annotations = {
            str(k): _copy_array(np.asarray(v), copy=copy) for k, v in ann_src.items()
        }
        self._mapping_meta = dict(mapping_meta or arrays.get("mapping_meta") or {})
        self._owner_id = owner_id if owner_id is not None else arrays.get("owner_id")
        self._units = units if units is not None else arrays.get("units")
        meta = units_meta if units_meta is not None else arrays.get("units_meta")
        self._units_meta = dict(meta or {})
        self._init_empty_indexes()
        self._rebuild_indexes()

    def _init_from_dataframe(
        self,
        df: pd.DataFrame,
        *,
        mapping_meta: dict[str, Any] | None,
        owner_id: Hashable | None,
        units: str | None,
        units_meta: Mapping[str, Any] | None,
    ) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Synapses table missing required columns: {missing}")

        n = len(df)
        synapse_id = np.asarray(df["synapse_id"].to_numpy(), dtype=object)
        if pd.Index(synapse_id).duplicated().any():
            dups = synapse_id[pd.Index(synapse_id).duplicated()].tolist()
            raise ValueError(f"Duplicate synapse_id values: {dups[:10]}")

        type_code = _type_codes_from_values(df["type"].to_numpy(), n)
        xyz = np.column_stack(
            [
                pd.to_numeric(df["x"], errors="raise").to_numpy(dtype=np.float64),
                pd.to_numeric(df["y"], errors="raise").to_numpy(dtype=np.float64),
                pd.to_numeric(df["z"], errors="raise").to_numpy(dtype=np.float64),
            ]
        )
        partner_id = np.asarray(df["partner_id"].to_numpy(), dtype=object)

        has_mapping = all(c in df.columns for c in MAPPING_COLUMNS)
        if has_mapping:
            edge_index = df["edge_index"].to_numpy(dtype=np.int64)
            edge_fraction = df["edge_fraction"].to_numpy(dtype=np.float64)
            distance_along_edge = df["distance_along_edge"].to_numpy(dtype=np.float64)
            nearest = np.column_stack(
                [
                    df["nearest_x"].to_numpy(dtype=np.float64),
                    df["nearest_y"].to_numpy(dtype=np.float64),
                    df["nearest_z"].to_numpy(dtype=np.float64),
                ]
            )
            distance_to_tree = df["distance_to_tree"].to_numpy(dtype=np.float64)
            mapped = df["mapped"].to_numpy(dtype=bool)
        else:
            edge_index = edge_fraction = distance_along_edge = None
            nearest = distance_to_tree = mapped = None

        reserved = set(REQUIRED_COLUMNS) | set(MAPPING_COLUMNS)
        annotations: dict[str, np.ndarray] = {}
        for col in df.columns:
            if col in reserved:
                continue
            annotations[str(col)] = np.asarray(df[col].to_numpy())

        self._synapse_id = synapse_id
        self._type_code = type_code
        self._xyz = xyz
        self._partner_id = partner_id
        self._edge_index = edge_index
        self._edge_fraction = edge_fraction
        self._distance_along_edge = distance_along_edge
        self._nearest = nearest
        self._distance_to_tree = distance_to_tree
        self._mapped = mapped
        self._annotations = annotations
        self._mapping_meta = dict(mapping_meta or {})
        self._owner_id = owner_id
        self._units = units
        self._units_meta = dict(units_meta or {})
        self._init_empty_indexes()
        self._rebuild_indexes()

    def _arrays_dict(self, *, copy: bool = False) -> dict[str, Any]:
        return {
            "synapse_id": self._synapse_id,
            "type_code": self._type_code,
            "xyz": self._xyz,
            "partner_id": self._partner_id,
            "edge_index": self._edge_index,
            "edge_fraction": self._edge_fraction,
            "distance_along_edge": self._distance_along_edge,
            "nearest": self._nearest,
            "distance_to_tree": self._distance_to_tree,
            "mapped": self._mapped,
            "annotations": self._annotations,
            "mapping_meta": self._mapping_meta,
            "owner_id": self._owner_id,
            "units": self._units,
            "units_meta": self._units_meta,
        }

    def _rebuild_indexes(self) -> None:
        """Rebuild runtime lookup indexes (not pickled)."""
        ids = self._synapse_id
        self._id_to_row = {ids[i]: i for i in range(len(ids))}

        partner_buckets: dict[Any, list[int]] = defaultdict(list)
        for i, pid in enumerate(self._partner_id):
            partner_buckets[pid].append(i)
        self._partner_to_rows = {
            k: np.asarray(v, dtype=np.int64) for k, v in partner_buckets.items()
        }

        if self._edge_index is None:
            self._edge_to_rows = {}
            return
        edge_buckets: dict[int, list[int]] = defaultdict(list)
        for i, e in enumerate(self._edge_index):
            edge_buckets[int(e)].append(i)
        self._edge_to_rows = {
            k: np.asarray(v, dtype=np.int64) for k, v in edge_buckets.items()
        }

    # --- constructors ---

    @classmethod
    def from_arrays(
        cls,
        *,
        synapse_id: Any = None,
        type: Any,  # noqa: A002
        coordinates: np.ndarray | None = None,
        x: Any = None,
        y: Any = None,
        z: Any = None,
        partner_id: Any = None,
        owner_id: Hashable | None = None,
        units: str | None = None,
        units_meta: Mapping[str, Any] | None = None,
        mapping_meta: dict[str, Any] | None = None,
        **extra_columns: Any,
    ) -> Self:
        """Build from NumPy-like arrays without requiring a DataFrame."""
        type_code = _type_codes_from_values(type)
        n = len(type_code)
        xyz = _coords_from_arrays(coordinates, x, y, z, n)
        if synapse_id is None:
            ids = np.arange(n, dtype=object)
        else:
            ids = np.asarray(_as_1d(synapse_id, "synapse_id", n), dtype=object)
        if pd.Index(ids).duplicated().any():
            dups = ids[pd.Index(ids).duplicated()].tolist()
            raise ValueError(f"Duplicate synapse_id values: {dups[:10]}")

        if partner_id is None:
            partners = _empty_object(n)
        else:
            partners = np.asarray(_as_1d(partner_id, "partner_id", n), dtype=object)

        annotations: dict[str, np.ndarray] = {}
        for key, values in extra_columns.items():
            if key in REQUIRED_COLUMNS or key in MAPPING_COLUMNS:
                raise ValueError(f"Duplicate / reserved column {key!r}")
            annotations[key] = np.asarray(_as_1d(values, key, n))

        return cls(
            _arrays={
                "synapse_id": ids,
                "type_code": type_code,
                "xyz": xyz,
                "partner_id": partners,
                "annotations": annotations,
            },
            copy=False,
            mapping_meta=mapping_meta,
            owner_id=owner_id,
            units=units,
            units_meta=units_meta,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Self:
        """Hydrate from a versioned SoA payload (or legacy v1 dict)."""
        obj = cls.__new__(cls)
        obj.__setstate__(dict(payload))
        return obj

    @classmethod
    def concat(cls, *parts: Synapses) -> Self:
        """Concatenate synapse tables; drops mapping; rejects duplicate IDs."""
        nonempty = [p for p in parts if p is not None and len(p) > 0]
        if not nonempty:
            if not parts:
                raise ValueError("concat requires at least one Synapses argument")
            first = parts[0]
            return cls(
                _arrays={
                    "synapse_id": np.asarray([], dtype=object),
                    "type_code": np.asarray([], dtype=np.uint8),
                    "xyz": np.zeros((0, 3), dtype=np.float64),
                    "partner_id": np.asarray([], dtype=object),
                    "annotations": {},
                },
                copy=False,
                owner_id=first._owner_id,
                units=first._units,
                units_meta=dict(first._units_meta),
            )

        ids = np.concatenate([p._synapse_id for p in nonempty])
        if pd.Index(ids).duplicated().any():
            dups = ids[pd.Index(ids).duplicated()].tolist()
            raise ValueError(f"Duplicate synapse_id values: {dups[:10]}")

        ann_keys: list[str] = []
        seen: set[str] = set()
        for p in nonempty:
            for k in p._annotations:
                if k not in seen:
                    seen.add(k)
                    ann_keys.append(k)

        annotations: dict[str, np.ndarray] = {}
        for key in ann_keys:
            chunks = []
            for p in nonempty:
                if key in p._annotations:
                    chunks.append(np.asarray(p._annotations[key]))
                else:
                    chunks.append(np.full(len(p), np.nan, dtype=object))
            annotations[key] = np.concatenate(chunks)

        first = nonempty[0]
        return cls(
            _arrays={
                "synapse_id": ids,
                "type_code": np.concatenate([p._type_code for p in nonempty]),
                "xyz": np.concatenate([p._xyz for p in nonempty], axis=0),
                "partner_id": np.concatenate([p._partner_id for p in nonempty]),
                "annotations": annotations,
            },
            copy=False,
            mapping_meta={},
            owner_id=first._owner_id,
            units=first._units,
            units_meta=dict(first._units_meta),
        )

    # --- pickle / graph-tool object gp ---

    def _to_payload(self) -> dict[str, Any]:
        """Versioned SoA dict suitable for graph-tool / ``.nr`` persistence."""
        arrays = {
            "synapse_id": np.asarray(self._synapse_id).copy(),
            "type_code": np.asarray(self._type_code, dtype=np.uint8).copy(),
            "xyz": np.asarray(self._xyz, dtype=np.float64).copy(),
            "partner_id": np.asarray(self._partner_id, dtype=object).copy(),
        }
        annotations = {k: np.asarray(v).copy() for k, v in self._annotations.items()}
        mapping: dict[str, Any] | None = None
        if self.is_mapped:
            mapping = {
                "edge_index": np.asarray(self._edge_index, dtype=np.int64).copy(),
                "edge_fraction": np.asarray(
                    self._edge_fraction, dtype=np.float64
                ).copy(),
                "distance_along_edge": np.asarray(
                    self._distance_along_edge, dtype=np.float64
                ).copy(),
                "nearest": np.asarray(self._nearest, dtype=np.float64).copy(),
                "distance_to_tree": np.asarray(
                    self._distance_to_tree, dtype=np.float64
                ).copy(),
                "mapped": np.asarray(self._mapped, dtype=bool).copy(),
            }
        return {
            "format": PAYLOAD_FORMAT,
            "version": PAYLOAD_VERSION,
            "arrays": arrays,
            "annotations": annotations,
            "mapping": mapping,
            "meta": {
                "mapping_meta": dict(self._mapping_meta),
                "owner_id": self._owner_id,
                "units": self._units,
                "units_meta": dict(self._units_meta),
            },
        }

    def __getstate__(self) -> dict[str, Any]:
        return self._to_payload()

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Legacy v1: {"df", "mapping_meta", "owner_id", "units", "units_meta"}
        if "df" in state and "format" not in state:
            df = state["df"]
            self._init_from_dataframe(
                df if isinstance(df, pd.DataFrame) else pd.DataFrame(df),
                mapping_meta=dict(state.get("mapping_meta") or {}),
                owner_id=state.get("owner_id"),
                units=state.get("units"),
                units_meta=dict(state.get("units_meta") or {}),
            )
            return

        version = int(state.get("version", PAYLOAD_VERSION))
        if version != PAYLOAD_VERSION and "arrays" not in state:
            raise ValueError(f"Unsupported synapses payload version {version}")

        arrays = dict(state["arrays"])
        mapping = state.get("mapping")
        if mapping:
            arrays.update(mapping)
        else:
            arrays.setdefault("edge_index", None)
        arrays["annotations"] = dict(state.get("annotations") or {})
        meta = dict(state.get("meta") or {})
        self._init_from_arrays(
            arrays,
            copy=False,
            mapping_meta=dict(
                meta.get("mapping_meta") or state.get("mapping_meta") or {}
            ),
            owner_id=meta.get("owner_id", state.get("owner_id")),
            units=meta.get("units", state.get("units")),
            units_meta=dict(meta.get("units_meta") or state.get("units_meta") or {}),
        )

    # --- basics ---

    def __len__(self) -> int:
        return len(self._synapse_id)

    def __repr__(self) -> str:
        n_pre = int(np.count_nonzero(self._type_code == TYPE_PRE))
        n_post = int(np.count_nonzero(self._type_code == TYPE_POST))
        mapped = "mapped" if self.is_mapped else "unmapped"
        owner = f", owner_id={self._owner_id!r}" if self._owner_id is not None else ""
        units = f", units={self._units!r}" if self._units is not None else ""
        return f"Synapses(n={len(self)}, pre={n_pre}, post={n_post}, {mapped}{owner}{units})"

    def copy(self) -> Self:
        """Deep-copy the table, mapping metadata, owner ID, and units."""
        return self.__class__(
            self,
            copy=True,
            mapping_meta=dict(self._mapping_meta),
            owner_id=self._owner_id,
            units=self._units,
            units_meta=dict(self._units_meta),
        )

    def to_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        """Materialize a pandas view of the SoA table."""
        data: dict[str, Any] = {
            "synapse_id": self._synapse_id,
            "type": self.types,
            "x": self._xyz[:, 0],
            "y": self._xyz[:, 1],
            "z": self._xyz[:, 2],
            "partner_id": self._partner_id,
        }
        for name, arr in self._annotations.items():
            data[name] = arr
        if self.is_mapped:
            data["edge_index"] = self._edge_index
            data["edge_fraction"] = self._edge_fraction
            data["distance_along_edge"] = self._distance_along_edge
            data["nearest_x"] = self._nearest[:, 0]
            data["nearest_y"] = self._nearest[:, 1]
            data["nearest_z"] = self._nearest[:, 2]
            data["distance_to_tree"] = self._distance_to_tree
            data["mapped"] = self._mapped
        df = pd.DataFrame(data)
        return df.copy() if copy else df

    @property
    def columns(self) -> list[str]:
        cols = list(REQUIRED_COLUMNS)
        cols.extend(self._annotations.keys())
        if self.is_mapped:
            cols.extend(MAPPING_COLUMNS)
        return cols

    @property
    def annotations(self) -> dict[str, np.ndarray]:
        """Extra annotation arrays keyed by column name."""
        return self._annotations

    @property
    def mapping_meta(self) -> dict[str, Any]:
        return self._mapping_meta

    @property
    def owner_id(self) -> Hashable | None:
        """Neuron / tree ID this table is bound to, or ``None`` if unbound."""
        return self._owner_id

    @owner_id.setter
    def owner_id(self, value: Hashable | None) -> None:
        self._owner_id = value

    @property
    def units(self) -> str | None:
        """Spatial units of raw coordinates, or ``None`` if unset."""
        return self._units

    @units.setter
    def units(self, value: str | None) -> None:
        self._units = value

    @property
    def units_meta(self) -> dict[str, Any]:
        """Unit metadata (e.g. voxel size/unit)."""
        return self._units_meta

    @units_meta.setter
    def units_meta(self, value: Mapping[str, Any] | None) -> None:
        self._units_meta = {} if value is None else dict(value)

    def set_units(
        self,
        units: str,
        *,
        voxel_size: float | None = None,
        voxel_unit: str | None = None,
    ) -> None:
        """Declare spatial units for synapse coordinates (does not rescale).

        Coordinates are assumed to already be expressed in *units*. Use tree
        ``convert_units`` after binding to rescale morphology + synapses together.
        """
        from ..utils.units import (
            VOXEL_SIZE_KEY,
            VOXEL_UNIT_KEY,
            VOXEL_UNITS,
            apply_voxel_metadata,
            is_voxel_units,
            normalize_units_str,
        )

        target = normalize_units_str(units)
        if is_voxel_units(target):
            if voxel_size is None or voxel_unit is None:
                raise ValueError("Voxel units require voxel_size and voxel_unit.")
            meta: dict[str, Any] = {}
            apply_voxel_metadata(meta, voxel_size, voxel_unit)
            self._units = VOXEL_UNITS
            self._units_meta = {
                VOXEL_SIZE_KEY: meta[VOXEL_SIZE_KEY],
                VOXEL_UNIT_KEY: meta[VOXEL_UNIT_KEY],
            }
        else:
            self._units = target
            self._units_meta = {}

    @property
    def is_mapped(self) -> bool:
        """True when a full mapping column set is present."""
        return (
            self._edge_index is not None
            and self._edge_fraction is not None
            and self._distance_along_edge is not None
            and self._nearest is not None
            and self._distance_to_tree is not None
            and self._mapped is not None
        )

    @property
    def mapping_valid(self) -> bool:
        """True when mapping columns exist and are marked valid."""
        return bool(self.is_mapped and self._mapping_meta.get("valid", False))

    # --- SoA accessors ---

    @property
    def synapse_ids(self) -> np.ndarray:
        return self._synapse_id

    @property
    def type_codes(self) -> np.ndarray:
        return self._type_code

    @property
    def types(self) -> np.ndarray:
        return _types_from_codes(self._type_code)

    @property
    def partner_ids(self) -> np.ndarray:
        return self._partner_id

    @property
    def xyz(self) -> np.ndarray:
        return self._xyz

    @property
    def edge_index(self) -> np.ndarray | None:
        return self._edge_index

    @property
    def edge_fraction(self) -> np.ndarray | None:
        return self._edge_fraction

    @property
    def distance_along_edge(self) -> np.ndarray | None:
        return self._distance_along_edge

    @property
    def nearest(self) -> np.ndarray | None:
        return self._nearest

    @property
    def distance_to_tree(self) -> np.ndarray | None:
        return self._distance_to_tree

    @property
    def mapped_flags(self) -> np.ndarray | None:
        return self._mapped

    # --- coordinates ---

    @property
    def coordinates(self) -> np.ndarray:
        """Raw synapse coordinates as ``(N, 3)`` float64."""
        return np.asarray(self._xyz, dtype=np.float64).copy()

    @property
    def mapped_coordinates(self) -> np.ndarray:
        """Nearest points on the morphology; NaN where unmapped."""
        if not self.is_mapped:
            raise ValueError("Synapses are not mapped; call tree.map_synapses() first")
        return np.asarray(self._nearest, dtype=np.float64).copy()

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

    # --- indexing / take ---

    def _column(self, name: str) -> np.ndarray:
        """Resolve a logical column to an array (core → mapping → annotations)."""
        if name == "synapse_id":
            return self._synapse_id
        if name == "type":
            return self.types
        if name == "x":
            return self._xyz[:, 0]
        if name == "y":
            return self._xyz[:, 1]
        if name == "z":
            return self._xyz[:, 2]
        if name == "partner_id":
            return self._partner_id
        if name == "edge_index":
            if self._edge_index is None:
                raise KeyError(name)
            return self._edge_index
        if name == "edge_fraction":
            if self._edge_fraction is None:
                raise KeyError(name)
            return self._edge_fraction
        if name == "distance_along_edge":
            if self._distance_along_edge is None:
                raise KeyError(name)
            return self._distance_along_edge
        if name == "nearest_x":
            if self._nearest is None:
                raise KeyError(name)
            return self._nearest[:, 0]
        if name == "nearest_y":
            if self._nearest is None:
                raise KeyError(name)
            return self._nearest[:, 1]
        if name == "nearest_z":
            if self._nearest is None:
                raise KeyError(name)
            return self._nearest[:, 2]
        if name == "distance_to_tree":
            if self._distance_to_tree is None:
                raise KeyError(name)
            return self._distance_to_tree
        if name == "mapped":
            if self._mapped is None:
                raise KeyError(name)
            return self._mapped
        if name in self._annotations:
            return self._annotations[name]
        raise KeyError(f"Unknown synapse column {name!r}")

    def _take_indices(self, indices: np.ndarray, *, copy: bool = False) -> Self:
        idx = np.asarray(indices, dtype=np.int64)
        arrays: dict[str, Any] = {
            "synapse_id": self._synapse_id[idx],
            "type_code": self._type_code[idx],
            "xyz": self._xyz[idx],
            "partner_id": self._partner_id[idx],
            "annotations": {k: v[idx] for k, v in self._annotations.items()},
        }
        if self.is_mapped:
            arrays["edge_index"] = self._edge_index[idx]
            arrays["edge_fraction"] = self._edge_fraction[idx]
            arrays["distance_along_edge"] = self._distance_along_edge[idx]
            arrays["nearest"] = self._nearest[idx]
            arrays["distance_to_tree"] = self._distance_to_tree[idx]
            arrays["mapped"] = self._mapped[idx]
        return self.__class__(
            _arrays=arrays,
            copy=copy,
            mapping_meta=dict(self._mapping_meta),
            owner_id=self._owner_id,
            units=self._units,
            units_meta=dict(self._units_meta),
        )

    def take(self, indices: Any) -> Self:
        """Return synapses at the given row indices."""
        return self._take_indices(np.asarray(indices, dtype=np.int64), copy=True)

    def mask(self, mask: Any) -> Self:
        """Return synapses where *mask* is True."""
        m = np.asarray(mask, dtype=bool)
        if m.shape != (len(self),):
            raise ValueError(f"mask length {m.shape[0]} != {len(self)}")
        return self._take_indices(np.flatnonzero(m), copy=True)

    def index_of_id(self, synapse_id: Any) -> int:
        """Return the row index for *synapse_id*."""
        try:
            return self._id_to_row[synapse_id]
        except KeyError as exc:
            raise KeyError(f"synapse_id {synapse_id!r} not found") from exc

    def indices_for_partner(self, partner_id: Any) -> np.ndarray:
        """Return row indices for synapses with the given partner."""
        return self._partner_to_rows.get(
            partner_id, np.asarray([], dtype=np.int64)
        ).copy()

    def indices_on_edge(self, edge_index: int) -> np.ndarray:
        """Return row indices mapped onto *edge_index*."""
        return self._edge_to_rows.get(
            int(edge_index), np.asarray([], dtype=np.int64)
        ).copy()

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
        mask = np.ones(len(self), dtype=bool)

        type_set = resolve_synapse_type_filter(type)
        if type_set is not None:
            codes = {_TYPE_STR_TO_CODE[t] for t in type_set}
            mask &= np.isin(self._type_code, list(codes))

        if partner_id is not None:
            mask &= self._isin_mask(self._partner_id, partner_id)

        if synapse_id is not None:
            mask &= self._isin_mask(self._synapse_id, synapse_id)

        if mapped is not None:
            if self._mapped is None:
                if mapped:
                    mask[:] = False
            else:
                mask &= self._mapped == bool(mapped)

        if max_distance is not None:
            if self._distance_to_tree is None:
                raise ValueError("max_distance requires mapped synapses")
            dist = self._distance_to_tree
            mask &= np.isfinite(dist) & (dist <= float(max_distance))

        for col, value in column_equals.items():
            try:
                arr = self._column(col)
            except KeyError as exc:
                raise KeyError(f"Unknown synapse column {col!r}") from exc
            mask &= self._isin_mask(arr, value)

        return self._take_indices(np.flatnonzero(mask), copy=False)

    @staticmethod
    def _isin_mask(arr: np.ndarray, value: Any) -> np.ndarray:
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            return arr == value
        values = list(value)
        # object / mixed equality
        mixed = arr.dtype == object or any(
            not isinstance(v, (int, float, np.number, str, bytes)) for v in values
        )
        if mixed:
            return np.array([a in values for a in arr], dtype=bool)
        return np.isin(arr, values)

    # --- mutation helpers (used by ops) ---

    def clear_mapping(self) -> None:
        """Drop derived mapping columns and mark mapping invalid."""
        self._clear_mapping_arrays()
        self._mapping_meta = {}
        self._edge_to_rows = {}

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
        n = len(self)
        edge_index = _as_1d(edge_index, "edge_index", n).astype(np.int64, copy=False)
        edge_fraction = _as_1d(edge_fraction, "edge_fraction", n).astype(
            np.float64, copy=False
        )
        distance_to_tree = _as_1d(distance_to_tree, "distance_to_tree", n).astype(
            np.float64, copy=False
        )
        mapped = _as_1d(mapped, "mapped", n).astype(bool, copy=False)
        nearest = np.asarray(nearest, dtype=np.float64)
        if nearest.shape != (n, 3):
            raise ValueError(f"nearest must have shape ({n}, 3), got {nearest.shape}")
        if distance_along_edge is None:
            distance_along_edge = np.full(n, np.nan, dtype=np.float64)
        else:
            distance_along_edge = _as_1d(
                distance_along_edge, "distance_along_edge", n
            ).astype(np.float64, copy=False)

        self._edge_index = np.asarray(edge_index, dtype=np.int64)
        self._edge_fraction = np.asarray(edge_fraction, dtype=np.float64)
        self._distance_along_edge = np.asarray(distance_along_edge, dtype=np.float64)
        self._nearest = np.ascontiguousarray(nearest, dtype=np.float64)
        self._distance_to_tree = np.asarray(distance_to_tree, dtype=np.float64)
        self._mapped = np.asarray(mapped, dtype=bool)
        self._mapping_meta = {
            "valid": True,
            "method": method,
            "version": MAPPING_VERSION,
            "max_distance": max_distance,
        }
        self._rebuild_indexes()

    def update_mapped_flags(
        self,
        mapped: np.ndarray,
        *,
        max_distance: float | None = None,
    ) -> None:
        """Update ``mapped`` flags in place (requires existing mapping)."""
        if not self.is_mapped:
            raise ValueError("Synapses are not mapped")
        mapped = _as_1d(mapped, "mapped", len(self)).astype(bool, copy=False)
        self._mapped = np.asarray(mapped, dtype=bool)
        self._mapping_meta = dict(self._mapping_meta)
        self._mapping_meta["valid"] = True
        if max_distance is not None:
            self._mapping_meta["max_distance"] = max_distance

    def remap_edges_through_reduction(self, reduction_map) -> None:
        """Transfer continuous edge locations through a :class:`ReductionMap`.

        Preserves raw coordinates, mapped coordinates, and ``distance_to_tree``.
        Updates ``edge_index``, ``edge_fraction``, and ``distance_along_edge`` to
        refer to reduced section edges (cable distance along the section).
        """
        if not self.is_mapped:
            return
        n = len(self)
        new_edges = np.empty(n, dtype=np.int64)
        new_fracs = np.empty(n, dtype=np.float64)
        new_dists = np.empty(n, dtype=np.float64)
        old_edges = self._edge_index
        fracs = self._edge_fraction
        locals_ = self._distance_along_edge

        for i in range(n):
            eidx = int(old_edges[i])
            try:
                new_e, offset, length, section_length = (
                    reduction_map.old_edge_to_section[eidx]
                )
            except KeyError as exc:
                raise KeyError(
                    f"Original edge {eidx} missing from reduction map"
                ) from exc

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

        self._edge_index = new_edges
        self._edge_fraction = new_fracs
        self._distance_along_edge = new_dists
        meta = dict(self._mapping_meta)
        meta["valid"] = True
        meta["method"] = "reduction_transfer"
        self._mapping_meta = meta
        self._rebuild_indexes()

    def remap_edge_indices(
        self,
        old_to_new: Mapping[int, int],
        *,
        mapped_only: bool = True,
    ) -> Self:
        """Keep synapses on retained edges and remap ``edge_index`` values.

        Returns a new :class:`Synapses` (or empty via caller clearing). Rows
        whose edge is not in *old_to_new* are dropped. When *mapped_only* is
        True, also require ``mapped=True``.
        """
        if not self.is_mapped:
            raise ValueError("remap_edge_indices requires mapped synapses")
        keep = np.array([int(e) in old_to_new for e in self._edge_index], dtype=bool)
        if mapped_only:
            keep &= self._mapped
        if not np.any(keep):
            return self._take_indices(np.asarray([], dtype=np.int64), copy=False)

        out = self._take_indices(np.flatnonzero(keep), copy=True)
        out._edge_index = np.asarray(
            [old_to_new[int(e)] for e in out._edge_index], dtype=np.int64
        )
        out._mapping_meta = dict(self._mapping_meta)
        out._mapping_meta["valid"] = True
        out._rebuild_indexes()
        return out

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
        coords = np.asarray(self._xyz, dtype=np.float64)
        m = None
        t = None
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
        self._xyz = np.ascontiguousarray(coords, dtype=np.float64)

        if self.is_mapped:
            nearest = np.asarray(self._nearest, dtype=np.float64)
            if m is not None:
                nearest = nearest @ m.T
            if scale is not None:
                nearest = nearest * float(scale)
                self._distance_to_tree = self._distance_to_tree * float(scale)
            if t is not None:
                nearest = nearest + t
            self._nearest = np.ascontiguousarray(nearest, dtype=np.float64)

    def apply_soa_transform(
        self,
        fn: Callable[
            [np.ndarray, np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ],
        *,
        distance_scale: float | None = None,
    ) -> None:
        """Apply an ``(x, y, z) → (x, y, z)`` SoA transform to raw (+ mapped) coords."""
        x, y, z = fn(
            self._xyz[:, 0].copy(), self._xyz[:, 1].copy(), self._xyz[:, 2].copy()
        )
        self._xyz = np.column_stack(
            [
                np.asarray(x, dtype=np.float64),
                np.asarray(y, dtype=np.float64),
                np.asarray(z, dtype=np.float64),
            ]
        )
        if self.is_mapped:
            nx, ny, nz = fn(
                self._nearest[:, 0].copy(),
                self._nearest[:, 1].copy(),
                self._nearest[:, 2].copy(),
            )
            self._nearest = np.column_stack(
                [
                    np.asarray(nx, dtype=np.float64),
                    np.asarray(ny, dtype=np.float64),
                    np.asarray(nz, dtype=np.float64),
                ]
            )
            if distance_scale is not None:
                self._distance_to_tree = self._distance_to_tree * float(distance_scale)

    def add_column(self, name: str, values: Any, *, overwrite: bool = False) -> None:
        """Attach an arbitrary annotation column."""
        if name in REQUIRED_COLUMNS or name in MAPPING_COLUMNS:
            raise ValueError(f"Cannot overwrite required/mapping column {name!r}")
        if name in self._annotations and not overwrite:
            raise ValueError(f"Column {name!r} already exists; pass overwrite=True")
        self._annotations[name] = np.asarray(_as_1d(values, name, len(self)))

    def summary(self) -> dict[str, Any]:
        """Compact counts useful for QC."""
        dist = None
        n_mapped = 0
        n_unmapped = len(self)
        if self.is_mapped:
            mapped_mask = self._mapped
            n_mapped = int(mapped_mask.sum())
            n_unmapped = int((~mapped_mask).sum())
            d = self._distance_to_tree[mapped_mask]
            dist = d if d.size else None

        out: dict[str, Any] = {
            "n_synapses": len(self),
            "n_pre": int(np.count_nonzero(self._type_code == TYPE_PRE)),
            "n_post": int(np.count_nonzero(self._type_code == TYPE_POST)),
            "n_mapped": n_mapped,
            "n_unmapped": n_unmapped,
            "median_distance_to_tree": (
                float(np.median(dist)) if dist is not None else None
            ),
            "max_distance_to_tree": float(np.max(dist)) if dist is not None else None,
            "mapping_valid": self.mapping_valid,
        }
        return out
