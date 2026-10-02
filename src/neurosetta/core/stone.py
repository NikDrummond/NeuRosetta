"""Base class — every singleton object inherits from this."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, Self

from .stone_helpers import (
    copy_metadata,
    normalize_stone_name,
    stone_eq,
    validate_stone_id,
)


class _Stone:
    """Core single-item class with identity and metadata.

    Subclasses such as :class:`~neurosetta.core.tree._Tree` may store
    ``ID``, ``name``, and ``metadata`` elsewhere (for example on
    ``graph.gp``). The slots declared here apply to direct ``_Stone`` /
    ``_Mesh`` instances.

    Identity
    --------
    * ``ID`` — logical object / neuron identifier (hashable; commonly
      ``int`` or ``str``). Binding key across representations.
    * ``name`` — artifact / display / default-filename string. Independent
      of ``ID``. Defaults to ``str(ID)`` when not supplied.
    """

    __slots__ = ("ID", "name", "metadata")

    def __init__(
        self,
        ID: Hashable,
        metadata: dict | None = None,
        *,
        name: str | None = None,
    ) -> None:
        self.ID = validate_stone_id(ID)
        self.name = normalize_stone_name(name, fallback_id=self.ID)
        self.metadata = {} if metadata is None else dict(metadata)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, ID={self.ID!r})"

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

    def list_meta(self, *, include_protected: bool = False) -> list[str]:
        """Return sorted metadata keys.

        ``include_protected`` is accepted for API parity with
        :class:`~neurosetta.core.tree._Tree`. Stones without a protected-key
        scheme return all keys regardless of the flag.
        """
        return sorted(self.metadata)

    def meta_summary(self, *, include_protected: bool = False) -> dict[str, int]:
        """Return ``{key: 1}`` for each metadata key defined on this object."""
        return {key: 1 for key in self.list_meta(include_protected=include_protected)}

    # --- copy ---

    def copy(self) -> Self:
        """Return a shallow copy with duplicated metadata dict."""
        return type(self)(
            ID=self.ID,
            metadata=copy_metadata(self.metadata),
            name=self.name,
        )

    clone = copy
