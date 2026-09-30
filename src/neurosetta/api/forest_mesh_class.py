"""Forest container for multiple neuron meshes (API).

This module provides the Forest_mesh class for managing collections of
Tree_mesh objects.
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
from .tree_mesh_class import Tree_mesh


class Forest_mesh(_Forest):
    """Batch container of neuron meshes (``Tree_mesh`` only) — I/O helper.

    Preferred workflow is :meth:`~neurosetta.api.Forest.set_meshes` to attach
    surfaces onto morphology :class:`~neurosetta.api.Tree` members. A
    ``Forest_mesh`` is what :func:`~neurosetta.import_mesh` returns for a
    directory of neuron meshes, and a transient batch for plotting — **not** a
    parallel :class:`~neurosetta.api.Forest` for analysis.

    Not a neuropil collection — use :class:`~neurosetta.api.Neuropils` for
    compartment boundaries. Members keep neuron ``mesh_kind`` / ID policy.

    Parameters
    ----------
    meshes : list[Tree_mesh]
        List of Tree_mesh objects to include in the collection.
    """

    __slots__ = ()

    def __init__(self, meshes: Iterable[Tree_mesh]) -> None:
        items = list(meshes)
        for m in items:
            if not isinstance(m, Tree_mesh):
                raise TypeError(f"Forest_mesh members must be Tree_mesh instances; got {type(m)!r}")
        super().__init__(trees=items)

    def build_3d(self, *args, **kwargs):
        """Not supported — use ``show_3d()`` or ``Viewer.add_mesh``."""
        raise NotImplementedError(
            "Forest_mesh.build_3d() is not supported; use show_3d() or "
            "Viewer.add_mesh(...) instead."
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
