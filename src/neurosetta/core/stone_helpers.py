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
    """Compare stones by type, ``ID``, and metadata contents."""
    if type(other) is not type(self):
        return NotImplemented
    return self.ID == other.ID and self.metadata == other.metadata


def validate_stone_id(ID: Hashable) -> Hashable:
    """Normalise and validate a stone ``ID``."""
    if isinstance(ID, bool):
        raise TypeError("Stone ID must not be a bool")
    return ID
