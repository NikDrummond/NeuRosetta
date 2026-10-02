"""Neuron Mesh class.

This module provides the Tree_mesh class for representing neuron morphologies
as surface meshes.
"""

from collections.abc import Hashable

from vedo import Mesh

from ..core import _Mesh
from ..core.mesh import MESH_KIND_NEURON
from ..io.mesh_utils import export_mesh
from ..ops.mesh import (
    count_faces,
    count_vertices,
    distance_to_surface,
    get_bounds,
    get_vertex_coordinates,
)
from ..ops.plotting.plot_mesh import plot_mesh
from ..ops.units import mesh_units as _mesh_units


class Tree_mesh(_Mesh):
    """Neuron morphology surface mesh (I/O + ``tree.mesh`` payload).

    Role
    ----
    Surface of a **single neuron**. Preferred workflow is to **attach** it to a
    morphology via :meth:`~neurosetta.api.Tree.set_mesh` /
    ``tree.mesh = …`` and work from the :class:`~neurosetta.api.Tree` (same
    pattern as ``tree.synapses``). Standalone ``Tree_mesh`` analysis as the
    primary neuron API is soft-deprecated; the class remains the import /
    export / facet value type.

    It is **not** a brain-compartment boundary — pass compartment geometry as
    :class:`~neurosetta.api.Neuropil` into
    :class:`~neurosetta.api.AnatomicalFrame` instead.

    Identity
    --------
    * ``ID`` — logical neuron identifier; must be compatible with the owning
      :class:`~neurosetta.api.Tree` ``ID`` when bound (same idea as
      ``Synapses.owner_id``). Use
      :func:`~neurosetta.core.mesh.check_neuron_mesh_owner_id` to enforce.
    * ``name`` — artifact / display / default-filename string. May differ from
      the Tree's ``name``; attachment does **not** overwrite ``name``.

    Parameters
    ----------
    ID : hashable
        Logical neuron identifier (align with ``Tree.ID`` for attachment).
    metadata : dict
        Metadata dictionary; ``mesh_kind=\"neuron\"`` is stamped on construct.
    mesh : vedo.Mesh
        Surface mesh of the neuron.
    name : str or None, optional
        Artifact name. Defaults to ``str(ID)``.
    """

    __slots__ = ()
    mesh_kind = MESH_KIND_NEURON

    def __init__(
        self,
        ID: Hashable,
        metadata: dict,
        mesh: Mesh,
        *,
        name: str | None = None,
    ) -> None:
        super().__init__(ID=ID, metadata=metadata, mesh=mesh, name=name)

    # --- geometry ---
    count_vertices = count_vertices
    count_faces = count_faces
    get_vertex_coordinates = get_vertex_coordinates
    get_bounds = get_bounds
    distance_to_surface = distance_to_surface

    # --- units ---
    get_units = _mesh_units.get_units
    get_voxel_spec = _mesh_units.get_voxel_spec
    set_units = _mesh_units.set_units
    set_voxel_units = _mesh_units.set_voxel_units
    convert_units = _mesh_units.convert_units
    snap_voxel_coordinates = _mesh_units.snap_voxel_coordinates
    check_units_defined = _mesh_units.check_units_defined

    # --- I/O / plotting ---
    export_mesh = export_mesh
    show_3d = plot_mesh
