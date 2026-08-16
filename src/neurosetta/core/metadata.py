"""Protected tree metadata keys and validation."""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping

from ..utils.units.voxel import VOXEL_SIZE_KEY, VOXEL_UNIT_KEY

PROTECTED_META_KEYS = frozenset(
    {
        "units",
        "file_path",
        "isReduced",
        "Flag",
        VOXEL_SIZE_KEY,
        VOXEL_UNIT_KEY,
    }
)

_PROTECTED_META_HINTS = {
    "units": "tree.set_units()",
    "file_path": "I/O on load/save",
    "isReduced": "tree.update_reduced()",
    "Flag": "tree.set_flag()",
    VOXEL_SIZE_KEY: "tree.set_voxel_units()",
    VOXEL_UNIT_KEY: "tree.set_voxel_units()",
}


def check_protected_meta_key(key: str) -> None:
    """Raise ``KeyError`` when ``key`` is a protected metadata entry."""
    if key in PROTECTED_META_KEYS:
        hint = _PROTECTED_META_HINTS.get(key, "the dedicated API")
        raise KeyError(f"metadata[{key!r}] is protected; use {hint}.")


def unwrap_metadata(meta: dict | _MetadataDict) -> dict:
    """Return the underlying plain dict for a metadata mapping."""
    if isinstance(meta, _MetadataDict):
        return meta._data
    return meta


def wrap_metadata(data: dict | _MetadataDict) -> _MetadataDict:
    """Wrap a plain dict as protected metadata (no-op if already wrapped)."""
    if isinstance(data, _MetadataDict):
        return data
    return _MetadataDict(dict(data))


def set_core_meta(meta: dict | _MetadataDict, key: str, value) -> None:
    """Write a protected core metadata entry (library/internal use)."""
    unwrap_metadata(meta)[key] = value


def del_core_meta(meta: dict | _MetadataDict, key: str) -> None:
    """Delete a metadata entry, including protected keys (library/internal use)."""
    unwrap_metadata(meta).pop(key, None)


class _MetadataDict(MutableMapping):
    """Metadata mapping that blocks user writes to protected core keys."""

    __slots__ = ("_data",)

    def __init__(self, data: dict | None = None) -> None:
        self._data: dict = {} if data is None else dict(data)

    def _check_write(self, key: str) -> None:
        check_protected_meta_key(key)

    def __getitem__(self, key: str):
        return self._data[key]

    def __setitem__(self, key: str, value) -> None:
        self._check_write(key)
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        self._check_write(key)
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return repr(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _MetadataDict):
            return self._data == other._data
        if isinstance(other, dict):
            return self._data == other
        return NotImplemented

    def copy(self) -> dict:
        """Return a shallow plain-dict copy of the metadata."""
        return self._data.copy()

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def setdefault(self, key: str, default=None):
        if key in self._data:
            return self._data[key]
        self._data[key] = default
        return default

    def pop(self, key: str, *default):
        self._check_write(key)
        if default:
            return self._data.pop(key, default[0])
        return self._data.pop(key)

    def popitem(self):
        for key in self._data:
            if key not in PROTECTED_META_KEYS:
                return key, self._data.pop(key)
        raise KeyError("no removable user metadata entries")

    def update(self, other=(), /, **kwargs) -> None:
        to_apply: dict = {}
        if other:
            if isinstance(other, _MetadataDict):
                to_apply.update(other._data)
            elif isinstance(other, dict):
                to_apply.update(other)
            else:
                to_apply.update(dict(other))
        to_apply.update(kwargs)
        for key in to_apply:
            self._check_write(key)
        self._data.update(to_apply)

    def clear(self) -> None:
        for key in list(self._data):
            if key not in PROTECTED_META_KEYS:
                del self._data[key]


__all__ = [
    "PROTECTED_META_KEYS",
    "_MetadataDict",
    "check_protected_meta_key",
    "del_core_meta",
    "set_core_meta",
    "unwrap_metadata",
    "wrap_metadata",
]
