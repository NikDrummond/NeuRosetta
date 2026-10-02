"""Helpers for :class:`~neurosetta.core.stone._Stone` identity and metadata."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stone import _Stone


def copy_metadata(metadata: dict) -> dict:
    """Return a shallow plain-dict copy of stone metadata."""
    return dict(metadata)


def stone_eq(self: _Stone, other: object) -> bool:
    """Compare stones by type, ``ID``, ``name``, and metadata contents."""
    if type(other) is not type(self):
        return NotImplemented
    return self.ID == other.ID and self.name == other.name and self.metadata == other.metadata


def validate_stone_id(ID: Hashable) -> Hashable:
    """Normalise and validate a stone ``ID``.

    Accepts any hashable scalar except ``bool``. Digit-strings are **not**
    coerced to ``int`` — ``\"00123\"`` remains ``\"00123\"``.
    """
    if isinstance(ID, bool):
        raise TypeError("Stone ID must not be a bool")
    hash(ID)  # raise TypeError early for unhashables
    return ID


def normalize_stone_name(name: str | None, *, fallback_id: Hashable) -> str:
    """Return artifact name, defaulting to ``str(fallback_id)`` when *name* is None."""
    if name is None:
        return str(fallback_id)
    text = str(name)
    if not text:
        raise ValueError("Stone name must be a non-empty string")
    return text
