"""Unit tracking and conversion for neuron / neuropil meshes."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from ...core import _Forest, _Mesh
from ...utils.units import (
    DEFAULT_UNITS,
    TARGET_MICROMETER,
    VOXEL_UNITS,
    is_dimensionless,
    is_voxel_units,
    normalize_units_str,
    scale_factor,
    units_are_equal,
    validate_voxel_metadata,
    voxel_spec_from_metadata,
)
from .tree_units import _commit_units_metadata, _pending_units_metadata


def _scale_mesh_geometry(mesh: _Mesh, factor: float) -> None:
    """Scale mesh vertex coordinates by a multiplicative factor."""
    verts = np.asarray(mesh.mesh.vertices, dtype=np.float64)
    mesh.mesh.vertices = verts * float(factor)


def get_units(mesh: _Mesh) -> str:
    """Return canonical spatial units for a mesh."""
    return normalize_units_str(mesh.metadata.get("units"))


def get_voxel_spec(mesh: _Mesh) -> tuple[float, str] | None:
    """Return ``(voxel_size, voxel_unit)`` when the mesh uses voxel coordinates."""
    return voxel_spec_from_metadata(mesh.metadata)


def set_units(
    mesh: _Mesh,
    units: str | None = None,
    *,
    convert: bool = False,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Set spatial units in mesh metadata, optionally rescaling vertices.

    Parameters
    ----------
    mesh : _Mesh
        Target mesh (``Tree_mesh`` or ``Neuropil``).
    units : str or None, optional
        Canonical unit string or ``\"voxel\"``. When omitted, both
        *voxel_size* and *voxel_unit* must be given.
    convert : bool, optional
        Rescale vertex coordinates when changing units. By default False
        (declare / re-tag only).
    voxel_size, voxel_unit
        Required when *units* is voxel-based.
    """
    if units is None:
        if voxel_size is None or voxel_unit is None:
            raise ValueError("set_units() requires units, or both voxel_size and voxel_unit.")
        units = VOXEL_UNITS

    old_meta = dict(mesh.metadata)
    current = normalize_units_str(old_meta.get("units"))
    pending, target = _pending_units_metadata(
        mesh.metadata,
        units,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )

    if (
        convert
        and not units_are_equal(current, target, old_meta, pending)
        and not is_dimensionless(current)
    ):
        factor = scale_factor(
            current,
            target,
            from_metadata=old_meta,
            to_metadata=pending,
        )
        _scale_mesh_geometry(mesh, factor)

    _commit_units_metadata(mesh, pending)


def set_voxel_units(
    mesh: _Mesh,
    voxel_size: float,
    voxel_unit: str,
) -> None:
    """Tag mesh coordinates as voxel indices with a cubic edge length."""
    set_units(
        mesh,
        VOXEL_UNITS,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )


def convert_units(
    mesh: _Mesh,
    target_units: str | None = None,
    *,
    in_place: bool = True,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
):
    """Convert mesh vertices to target units.

    Raises
    ------
    ValueError
        If the mesh is dimensionless, or voxel kwargs are incomplete.
    """
    if target_units is None:
        if voxel_size is None or voxel_unit is None:
            raise ValueError(
                "convert_units() requires target_units, or both voxel_size and voxel_unit."
            )
        target_units = VOXEL_UNITS

    if not in_place:
        mesh = mesh.copy()

    old_meta = dict(mesh.metadata)
    current = normalize_units_str(old_meta.get("units"))
    if is_dimensionless(current):
        raise ValueError(
            "Cannot convert from dimensionless units; use mesh.set_units() "
            "to assign spatial units first."
        )
    pending, target = _pending_units_metadata(
        mesh.metadata,
        target_units,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )

    if not units_are_equal(current, target, old_meta, pending):
        factor = scale_factor(
            current,
            target,
            from_metadata=old_meta,
            to_metadata=pending,
        )
        _scale_mesh_geometry(mesh, factor)

    _commit_units_metadata(mesh, pending)
    return mesh


_VoxelSnapMethod = Literal["floor", "round", "ceil"]


def snap_voxel_coordinates(
    mesh: _Mesh,
    *,
    method: _VoxelSnapMethod = "floor",
) -> _Mesh:
    """Snap mesh vertices to integer voxel grid indices."""
    if not is_voxel_units(get_units(mesh)):
        raise ValueError(
            "snap_voxel_coordinates() requires voxel units; convert to voxels before snapping."
        )
    validate_voxel_metadata(mesh.metadata)

    if method not in ("floor", "round", "ceil"):
        raise ValueError("method must be 'floor', 'round', or 'ceil'.")

    verts = np.asarray(mesh.mesh.vertices, dtype=np.float64)
    if method == "floor":
        snapped = np.floor(verts)
    elif method == "round":
        snapped = np.round(verts)
    else:
        snapped = np.ceil(verts)

    mesh.mesh.vertices = snapped
    return mesh


def check_units_defined(mesh: _Mesh) -> None:
    """Raise if the mesh has dimensionless units or invalid voxel metadata."""
    if is_dimensionless(get_units(mesh)):
        raise ValueError("Mesh units are dimensionless; assign spatial units before converting.")
    if is_voxel_units(get_units(mesh)):
        validate_voxel_metadata(mesh.metadata)


def _defined_unit_spec(mesh: _Mesh) -> tuple[str, tuple[float, str] | None]:
    units = get_units(mesh)
    if is_voxel_units(units):
        return units, validate_voxel_metadata(mesh.metadata)
    return units, None


def harmonize_forest_units(
    forest: _Forest,
    target_units: str = TARGET_MICROMETER,
) -> None:
    """Convert all meshes in a collection to a common target unit."""
    target = normalize_units_str(target_units)
    defined = [_defined_unit_spec(m) for m in forest if not is_dimensionless(get_units(m))]

    if len(set(defined)) > 1:
        warnings.warn(
            f"Mesh collection contains mixed units "
            f"{[str(spec) for spec in set(defined)]}; "
            f"converting defined meshes to {target!r}.",
            UserWarning,
            stacklevel=2,
        )

    for m in forest:
        current = get_units(m)
        if is_dimensionless(current):
            warnings.warn(
                f"Mesh {m.ID} has dimensionless units; skipping conversion.",
                UserWarning,
                stacklevel=2,
            )
            continue
        convert_units(m, target, in_place=True)


def ensure_forest_units(
    forest: _Forest,
    target_units: str = TARGET_MICROMETER,
) -> None:
    """Harmonize collection units and require spatial units on every mesh."""
    harmonize_forest_units(forest, target_units=target_units)
    dimensionless_ids = [m.ID for m in forest if is_dimensionless(get_units(m))]
    if dimensionless_ids:
        raise ValueError(
            "Mesh collection contains meshes with dimensionless units: "
            f"{dimensionless_ids}. Assign units before running spatial "
            "mesh operations."
        )


__all__ = [
    "DEFAULT_UNITS",
    "get_units",
    "get_voxel_spec",
    "set_units",
    "set_voxel_units",
    "convert_units",
    "snap_voxel_coordinates",
    "check_units_defined",
    "harmonize_forest_units",
    "ensure_forest_units",
]
