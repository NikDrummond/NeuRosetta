"""Base class — every singleton object inherits from this."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Self

from .stone_helpers import copy_metadata, stone_eq, validate_stone_id


class _Stone:
    """Core single-item class with identity and metadata.

    Subclasses such as :class:`~neurosetta.core.tree._Tree` may store
    ``ID`` and ``metadata`` elsewhere (for example on ``graph.gp``). The
    slots declared here apply to direct ``_Stone`` / ``_Mesh`` instances.
    """

    __slots__ = ("ID", "metadata")

    def __init__(self, ID: Hashable, metadata: dict | None = None) -> None:
        self.ID = validate_stone_id(ID)
        self.metadata = {} if metadata is None else dict(metadata)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(ID={self.ID!r})"

    def __eq__(self, other: object) -> bool:
        return stone_eq(self, other)

    # --- user metadata ---

    def get_meta(self, key: str, default=None) -> Any:
        """Return a metadata value, or ``default`` if the key is absent."""
        return self.metadata.get(key, default)

    def has_meta(self, key: str) -> bool:
        """Return True when ``key`` exists in metadata."""
        return key in self.metadata

    def set_meta(self, key: str, value) -> None:
        """Set a metadata entry on the plain metadata dict."""
        self.metadata[key] = value

    def del_meta(self, key: str) -> None:
        """Delete a metadata entry."""
        del self.metadata[key]

    def list_meta(self) -> list[str]:
        """Return sorted metadata keys."""
        return sorted(self.metadata)

    def meta_summary(self) -> dict[str, int]:
        """Return ``{key: 1}`` for each metadata key defined on this object."""
        return {key: 1 for key in self.list_meta()}

    # --- copy ---

    def copy(self) -> Self:
        """Return a shallow copy with duplicated metadata dict."""
        return type(self)(ID=self.ID, metadata=copy_metadata(self.metadata))

    clone = copy
