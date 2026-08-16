"""Base mesh class."""

from collections.abc import Hashable

from vedo import Mesh

from .stone import _Stone


class _Mesh(_Stone):
    """Core mesh class."""

    __slots__ = ("mesh",)

    def __init__(self, ID: Hashable, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID, metadata)
        self.mesh = mesh

    def copy(self):
        """Return a shallow copy with duplicated metadata and mesh."""
        return type(self)(
            ID=self.ID,
            metadata=dict(self.metadata),
            mesh=self.mesh.clone() if hasattr(self.mesh, "clone") else self.mesh,
        )

    clone = copy
