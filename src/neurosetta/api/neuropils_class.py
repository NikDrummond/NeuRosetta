"""Neuropil container class.

This module provides the Neuropils class for managing collections of Neuropil
objects.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..core import _Forest
from ..io.mesh_utils import export_mesh
from ..ops.mesh import (
    count_faces,
    count_vertices,
    get_bounds,
    get_vertex_coordinates,
)
from ..ops.plotting.plot_mesh import plot_mesh
from ..ops.units import mesh_units as _mesh_units
from .forest_class import _forest_op
from .neuropil_class import Neuropil


class Neuropils(_Forest):
    """API collection of compartment / neuropil meshes (``Neuropil`` only).

    Not a neuron-mesh collection — use :class:`~neurosetta.api.Forest_mesh`
    for morphology surfaces.

    Parameters
    ----------
    meshes : list[Neuropil]
        List of Neuropil objects to include in the collection.
    """

    __slots__ = ()

    def __init__(self, meshes: Iterable[Neuropil]) -> None:
        items = list(meshes)
        for m in items:
            if not isinstance(m, Neuropil):
                raise TypeError(f"Neuropils members must be Neuropil instances; got {type(m)!r}")
        super().__init__(trees=items)

    def build_3d(self, *args, **kwargs):
        """Not supported — use ``show_3d()`` or ``Viewer.add_mesh``."""
        raise NotImplementedError(
            "Neuropils.build_3d() is not supported; use show_3d() or Viewer.add_mesh(...) instead."
        )

    # --- geometry (per member) ---
    count_vertices = _forest_op(count_vertices)
    count_faces = _forest_op(count_faces)
    get_vertex_coordinates = _forest_op(get_vertex_coordinates)
    get_bounds = _forest_op(get_bounds)

    # --- units ---
    get_units = _forest_op(_mesh_units.get_units)
    set_units = _forest_op(_mesh_units.set_units)
    convert_units = _forest_op(_mesh_units.convert_units)
    snap_voxel_coordinates = _forest_op(_mesh_units.snap_voxel_coordinates)
    harmonize_forest_units = _mesh_units.harmonize_forest_units
    ensure_forest_units = _mesh_units.ensure_forest_units

    # --- I/O / plotting ---
    export_mesh = export_mesh
    show_3d = plot_mesh
