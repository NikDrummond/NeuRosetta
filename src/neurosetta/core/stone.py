"""Base class — every singleton object inherits from this."""

from collections.abc import Hashable


class _Stone:
    """Core single-item class with identity and metadata."""

    __slots__ = ("ID", "metadata")

    def __init__(self, ID: Hashable, metadata: dict) -> None:
        self.ID = ID
        self.metadata = metadata
