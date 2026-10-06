"""Serializer registry for Workspace analysis artifacts.

Supported v1 artifact types use stable string identifiers (``json``,
``numpy``, ``dataframe``) rather than Python module paths so the on-disk
contract can evolve independently of import layout.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from .workspace_errors import WorkspaceFormatError

# Stable on-disk type identifiers (not Python qualnames).
JSON_TYPE = "json"
NUMPY_TYPE = "numpy"
DATAFRAME_TYPE = "dataframe"

SUPPORTED_RESULT_TYPES = (JSON_TYPE, NUMPY_TYPE, DATAFRAME_TYPE)


def validate_json_compatible(value: Any, *, path: str = "value") -> None:
    """Raise ``TypeError`` if *value* is not JSON-serializable.

    Parameters
    ----------
    value
        Object to validate.
    path : str, optional
        Human-readable location used in error messages (e.g.
        ``\"metadata['foo']['bar']\"``).

    Raises
    ------
    TypeError
        If *value* (or a nested element) cannot be represented in JSON
        without lossy conversion.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise TypeError(f"{path} is not JSON serializable: non-finite float ({value!r})")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} dict keys must be str; got {type(key).__name__} ({key!r})")
            validate_json_compatible(item, path=f"{path}[{key!r}]")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_json_compatible(item, path=f"{path}[{i}]")
        return
    raise TypeError(f"{path} is not JSON serializable: {type(value).__name__}")


class ArtifactSerializer(ABC):
    """Serialize / deserialize one supported Workspace artifact type."""

    type_id: str
    extension: str

    @abstractmethod
    def can_serialize(self, value: Any) -> bool:
        """Return True when *value* is handled by this serializer."""

    @abstractmethod
    def dump(self, value: Any, path: Path) -> None:
        """Write *value* to *path*."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Read a value previously written by :meth:`dump`."""


class JsonSerializer(ArtifactSerializer):
    """JSON-compatible Python scalars and nested containers."""

    type_id = JSON_TYPE
    extension = ".json"

    def can_serialize(self, value: Any) -> bool:
        if isinstance(value, (np.ndarray, pd.DataFrame)):
            return False
        try:
            validate_json_compatible(value, path="result")
        except TypeError:
            return False
        return True

    def dump(self, value: Any, path: Path) -> None:
        validate_json_compatible(value, path="result")
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def load(self, path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8"))


class NumpySerializer(ArtifactSerializer):
    """NumPy ``ndarray`` via ``np.save`` / ``np.load`` (no pickle)."""

    type_id = NUMPY_TYPE
    extension = ".npy"

    def can_serialize(self, value: Any) -> bool:
        return isinstance(value, np.ndarray) and value.dtype != object

    def dump(self, value: Any, path: Path) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"expected ndarray, got {type(value).__name__}")
        if value.dtype == object:
            raise TypeError(
                "NumPy object arrays are not supported in Workspace artifacts (allow_pickle=False)"
            )
        np.save(path, value, allow_pickle=False)

    def load(self, path: Path) -> Any:
        return np.load(path, allow_pickle=False)


class DataFrameSerializer(ArtifactSerializer):
    """pandas ``DataFrame`` via table-oriented JSON.

    Notes
    -----
    Uses ``orient=\"table\"``, which round-trips column names, dtypes, and a
    simple Index for typical numeric / string tables. Complex indexes
    (``MultiIndex``), categorical dtypes with non-JSON categories, and
    extension arrays outside pandas' table schema are not guaranteed.
    """

    type_id = DATAFRAME_TYPE
    extension = ".json"

    def can_serialize(self, value: Any) -> bool:
        return isinstance(value, pd.DataFrame)

    def dump(self, value: Any, path: Path) -> None:
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"expected DataFrame, got {type(value).__name__}")
        path.write_text(value.to_json(orient="table", date_format="iso") + "\n", encoding="utf-8")

    def load(self, path: Path) -> Any:
        return pd.read_json(path, orient="table")


_REGISTRY: tuple[ArtifactSerializer, ...] = (
    NumpySerializer(),
    DataFrameSerializer(),
    JsonSerializer(),
)

_BY_TYPE: dict[str, ArtifactSerializer] = {s.type_id: s for s in _REGISTRY}


def serializer_for(value: Any) -> ArtifactSerializer:
    """Return the serializer that handles *value*.

    Raises
    ------
    TypeError
        If no registered serializer accepts *value*.
    """
    for serializer in _REGISTRY:
        if serializer.can_serialize(value):
            return serializer
    raise TypeError(
        f"unsupported Workspace result type {type(value).__name__}; "
        f"supported artifact types: {', '.join(SUPPORTED_RESULT_TYPES)}"
    )


def serializer_for_type(type_id: str) -> ArtifactSerializer:
    """Return the serializer registered under *type_id*.

    Raises
    ------
    WorkspaceFormatError
        If *type_id* is unknown.
    """
    try:
        return _BY_TYPE[type_id]
    except KeyError as exc:
        raise WorkspaceFormatError(
            f"unknown Workspace artifact type {type_id!r}; "
            f"supported: {', '.join(SUPPORTED_RESULT_TYPES)}"
        ) from exc


__all__ = [
    "DATAFRAME_TYPE",
    "JSON_TYPE",
    "NUMPY_TYPE",
    "SUPPORTED_RESULT_TYPES",
    "ArtifactSerializer",
    "DataFrameSerializer",
    "JsonSerializer",
    "NumpySerializer",
    "serializer_for",
    "serializer_for_type",
    "validate_json_compatible",
]
