"""Neuropil mesh class.

This module provides the Neuropil class for representing brain region
boundaries as surface meshes.
"""

from vedo import Mesh

from ..core import _Mesh
from ..core.mesh import MESH_KIND_NEUROPIL
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


class Neuropil(_Mesh):
    """Brain-compartment / region boundary surface mesh.

    Role
    ----
    Anatomical **reference geometry** (neuropil, layer, ROI shell, …). Produced
    by :func:`~neurosetta.reconstruct_neuropil_surface` or
    ``import_mesh(..., mesh_type=\"Neuropil\")``. Valid input for
    :class:`~neurosetta.api.AnatomicalFrame` ``reference_mesh`` /
    ``inner_surface`` / ``outer_surface``.

    Not a neuron morphology surface — that is
    :class:`~neurosetta.api.Tree_mesh`.

    ID policy
    ---------
    ``ID`` is a region / compartment label (typically ``str``, e.g.
    ``\"AL\"``, ``\"lobula_plate\"``). It does **not** need to match a Tree ID.

    Parameters
    ----------
    ID : str
        Region / compartment identifier.
    metadata : dict
        Metadata dictionary; ``mesh_kind=\"neuropil\"`` is stamped on construct.
    mesh : vedo.Mesh
        Surface mesh of the compartment boundary.
    name : str or None, optional
        Artifact / display / default-filename string. Defaults to ``str(ID)``.
    """

    __slots__ = ()
    mesh_kind = MESH_KIND_NEUROPIL

    def __init__(
        self,
        ID: str,
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
