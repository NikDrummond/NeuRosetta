"""Unit checks and stamping for neuron meshes bound to trees."""

from __future__ import annotations

import warnings
from typing import Any

from ...core import _Tree
from ...utils.units import (
    is_dimensionless,
    is_voxel_units,
    units_are_equal,
)
from .mesh_units import get_units as get_mesh_units
from .tree_units import get_units, get_voxel_spec


def mesh_units_unset(mesh) -> bool:
    """True when the mesh has dimensionless / undeclared spatial units."""
    return is_dimensionless(get_mesh_units(mesh))


def tree_units_unset(tree: _Tree) -> bool:
    """True when the tree has dimensionless / undeclared spatial units."""
    return is_dimensionless(get_units(tree))


def _mesh_units_meta(mesh) -> dict[str, Any]:
    return dict(mesh.metadata)


def _tree_units_meta(tree: _Tree) -> dict[str, Any]:
    return dict(tree.metadata)


def check_mesh_tree_units(
    mesh,
    tree: _Tree,
    *,
    context: str = "operation",
) -> None:
    """Warn on unset units; raise when both sides are set and disagree."""
    mesh_unset = mesh_units_unset(mesh)
    tree_unset = tree_units_unset(tree)

    if mesh_unset and tree_unset:
        warnings.warn(
            f"{context}: mesh and tree {tree.ID!r} both lack spatial units; "
            f"assign units before relying on joint geometry.",
            UserWarning,
            stacklevel=2,
        )
        return

    if mesh_unset:
        warnings.warn(
            f"{context}: mesh has no units; tree {tree.ID!r} uses "
            f"{get_units(tree)!r}. Mesh coordinates will be treated as "
            f"matching the tree.",
            UserWarning,
            stacklevel=2,
        )
        return

    if tree_unset:
        warnings.warn(
            f"{context}: tree {tree.ID!r} has dimensionless units while "
            f"mesh declares {get_mesh_units(mesh)!r}. Assign tree units "
            f"before attaching.",
            UserWarning,
            stacklevel=2,
        )
        return

    if not units_are_equal(
        get_mesh_units(mesh),
        get_units(tree),
        _mesh_units_meta(mesh),
        _tree_units_meta(tree),
    ):
        raise ValueError(
            f"{context}: mesh units {get_mesh_units(mesh)!r} are incompatible "
            f"with tree {tree.ID!r} units {get_units(tree)!r}. Convert one "
            f"side before binding."
        )


def stamp_mesh_units_from_tree(mesh, tree: _Tree) -> None:
    """Copy the tree's canonical units onto *mesh* (including voxel metadata)."""
    from .mesh_units import set_units as set_mesh_units

    units = get_units(tree)
    if is_voxel_units(units):
        spec = get_voxel_spec(tree)
        if spec is None:
            set_mesh_units(mesh, units, convert=False)
        else:
            size, unit = spec
            set_mesh_units(
                mesh,
                units,
                convert=False,
                voxel_size=size,
                voxel_unit=unit,
            )
    else:
        set_mesh_units(mesh, units, convert=False)


__all__ = [
    "mesh_units_unset",
    "tree_units_unset",
    "check_mesh_tree_units",
    "stamp_mesh_units_from_tree",
]
