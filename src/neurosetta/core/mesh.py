"""Base mesh class and mesh-kind taxonomy."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, ClassVar, Self

from vedo import Mesh

from .stone import _Stone

# Canonical mesh roles (also stamped on ``metadata["mesh_kind"]``).
MESH_KIND_KEY = "mesh_kind"
MESH_KIND_GENERIC = "mesh"
MESH_KIND_NEURON = "neuron"
MESH_KIND_NEUROPIL = "neuropil"


def mesh_kind_of(obj: Any) -> str:
    """Return the taxonomy kind for a mesh-like object.

    Prefers the class ``mesh_kind`` attribute, then ``metadata["mesh_kind"]``,
    else :data:`MESH_KIND_GENERIC`.
    """
    kind = getattr(type(obj), "mesh_kind", None)
    if isinstance(kind, str) and kind:
        return kind
    meta = getattr(obj, "metadata", None)
    if isinstance(meta, dict):
        stamped = meta.get(MESH_KIND_KEY)
        if isinstance(stamped, str) and stamped:
            return stamped
    return MESH_KIND_GENERIC


def is_neuron_mesh(obj: Any) -> bool:
    """True when *obj* is a neuron morphology surface (``Tree_mesh``)."""
    return mesh_kind_of(obj) == MESH_KIND_NEURON


def is_neuropil_mesh(obj: Any) -> bool:
    """True when *obj* is a compartment / region boundary mesh."""
    return mesh_kind_of(obj) == MESH_KIND_NEUROPIL


def check_neuron_mesh_owner_id(mesh: _Mesh, owner_id: Hashable) -> None:
    """Require ``mesh.ID == owner_id`` for future ``tree.mesh`` attachment.

    Parameters
    ----------
    mesh : _Mesh
        Expected neuron mesh (``Tree_mesh``).
    owner_id : hashable
        Owning morphology ``Tree.ID``.

    Raises
    ------
    TypeError
        If *mesh* is not a neuron mesh.
    ValueError
        If IDs disagree.
    """
    if not is_neuron_mesh(mesh):
        raise TypeError(
            f"owner-ID check requires a neuron mesh (mesh_kind={MESH_KIND_NEURON!r}); "
            f"got mesh_kind={mesh_kind_of(mesh)!r} ({type(mesh).__name__})"
        )
    if owner_id != mesh.ID:
        raise ValueError(f"neuron mesh ID {mesh.ID!r} does not match owner ID {owner_id!r}")


class _Mesh(_Stone):
    """Core mesh class wrapping a :class:`vedo.Mesh`.

    Meshes are Stones (``ID`` + ``name`` + ``metadata``) and are **not**
    morphology graphs — graph-oriented Forest helpers such as
    ``list_properties`` / ``build_3d`` therefore return empty / raise clearly
    rather than pretending to be Trees.

    Identity
    --------
    * ``ID`` — logical identifier (for neuron meshes: owning ``Tree.ID``).
    * ``name`` — artifact / display / default-filename string. Independent of
      ``ID``. Defaults to ``str(ID)`` when not supplied.

    Taxonomy
    --------
    Subclasses declare ``mesh_kind``:

    * ``\"neuron\"`` — :class:`~neurosetta.api.Tree_mesh` (morphology surface)
    * ``\"neuropil\"`` — :class:`~neurosetta.api.Neuropil` (compartment boundary)
    * ``\"mesh\"`` — generic / low-level ``_Mesh``

    :class:`~neurosetta.api.AnatomicalFrame` accepts neuropil / generic meshes
    as reference geometry, **not** neuron meshes.
    """

    __slots__ = ("mesh",)
    mesh_kind: ClassVar[str] = MESH_KIND_GENERIC

    def __init__(
        self,
        ID: Hashable,
        metadata: dict,
        mesh: Mesh,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(ID, metadata, name=name)
        self.metadata.setdefault(MESH_KIND_KEY, type(self).mesh_kind)
        self.mesh = mesh

    def list_properties(self, level: str = "all") -> list:
        """Meshes have no graph-tool properties; return an empty list."""
        return []

    def copy(self) -> Self:
        """Return a shallow copy with duplicated metadata and mesh."""
        return type(self)(
            ID=self.ID,
            metadata=dict(self.metadata),
            mesh=self.mesh.clone() if hasattr(self.mesh, "clone") else self.mesh,
            name=self.name,
        )

    clone = copy

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, ID={self.ID!r})"
