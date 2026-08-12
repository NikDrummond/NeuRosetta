"""Base mesh class."""

from typing import Hashable

from vedo import Mesh

from .stone import _Stone


class _Mesh(_Stone):
    """Core mesh class."""

    __slots__ = ("mesh",)

    def __init__(self, ID: Hashable, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID, metadata)
        self.mesh = mesh
