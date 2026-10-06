"""Unit tracking, conversion, and tree-binding checks for synapse tables."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from ...core import _Tree
from ...core.synapses import Synapses
from ...utils.units import (
    VOXEL_SIZE_KEY,
    VOXEL_UNIT_KEY,
    VOXEL_UNITS,
    is_dimensionless,
    is_voxel_units,
    normalize_units_str,
    scale_factor,
    units_are_equal,
    validate_voxel_metadata,
    voxel_spec_from_metadata,
)
from .tree_units import (
    _pending_units_metadata,
)
from .tree_units import (
    get_units as get_tree_units,
)
from .tree_units import (
    get_voxel_spec as get_tree_voxel_spec,
)

_VoxelSnapMethod = Literal["floor", "round", "ceil"]


def synapse_units_unset(syn: Synapses) -> bool:
    """True when the synapse table has no declared spatial units."""
    return syn.units is None


def tree_units_unset(tree: _Tree) -> bool:
    """True when the tree has dimensionless / undeclared spatial units."""
    return is_dimensionless(get_tree_units(tree))


def _synapse_meta_view(syn: Synapses) -> dict[str, Any]:
    """Metadata dict compatible with tree/mesh unit helpers."""
    meta = dict(syn.units_meta)
    meta["units"] = syn.units
    return meta


def _synapse_units_meta(syn: Synapses) -> dict[str, Any]:
    return dict(syn.units_meta)


def _tree_units_meta(tree: _Tree) -> dict[str, Any]:
    return dict(tree.metadata)


def _commit_synapse_units(syn: Synapses, pending: dict[str, Any]) -> None:
    """Write pending unit metadata onto a synapse table."""
    syn._units = pending["units"]
    if is_voxel_units(pending["units"]):
        syn._units_meta = {
            VOXEL_SIZE_KEY: pending[VOXEL_SIZE_KEY],
            VOXEL_UNIT_KEY: pending[VOXEL_UNIT_KEY],
        }
    else:
        syn._units_meta = {}


def _scale_synapse_geometry(syn: Synapses, factor: float) -> None:
    """Scale raw (+ mapped) coordinates and length-like mapping fields."""
    syn.transform_coordinates(scale=float(factor))
    if syn.is_mapped and syn._distance_along_edge is not None:
        syn._distance_along_edge = np.asarray(
            syn._distance_along_edge * float(factor), dtype=np.float64
        )


def get_units(syn: Synapses) -> str:
    """Return canonical spatial units for a synapse table."""
    return normalize_units_str(syn.units)


def get_voxel_spec(syn: Synapses) -> tuple[float, str] | None:
    """Return ``(voxel_size, voxel_unit)`` when the table uses voxel coordinates."""
    return voxel_spec_from_metadata(_synapse_meta_view(syn))


def set_units(
    syn: Synapses,
    units: str | None = None,
    *,
    convert: bool = False,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Set spatial units on a synapse table, optionally rescaling coordinates.

    Parameters
    ----------
    syn : Synapses
        Target synapse table.
    units : str or None, optional
        Canonical unit string or ``\"voxel\"``. When omitted, both
        *voxel_size* and *voxel_unit* must be given.
    convert : bool, optional
        Rescale coordinates (and mapped distances) when changing units.
        By default False (declare / re-tag only).
    voxel_size, voxel_unit
        Required when *units* is voxel-based.
    """
    if units is None:
        if voxel_size is None or voxel_unit is None:
            raise ValueError("set_units() requires units, or both voxel_size and voxel_unit.")
        units = VOXEL_UNITS

    old_meta = _synapse_meta_view(syn)
    current = normalize_units_str(old_meta.get("units"))
    pending, target = _pending_units_metadata(
        old_meta,
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
        _scale_synapse_geometry(syn, factor)

    _commit_synapse_units(syn, pending)


def set_voxel_units(
    syn: Synapses,
    voxel_size: float,
    voxel_unit: str,
) -> None:
    """Tag synapse coordinates as voxel indices with a cubic edge length."""
    set_units(
        syn,
        VOXEL_UNITS,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )


def convert_units(
    syn: Synapses,
    target_units: str | None = None,
    *,
    in_place: bool = True,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> Synapses:
    """Convert synapse coordinates to target units.

    Raises
    ------
    ValueError
        If the table is dimensionless / unset, or voxel kwargs are incomplete.
    """
    if target_units is None:
        if voxel_size is None or voxel_unit is None:
            raise ValueError(
                "convert_units() requires target_units, or both voxel_size and voxel_unit."
            )
        target_units = VOXEL_UNITS

    if not in_place:
        syn = syn.copy()

    old_meta = _synapse_meta_view(syn)
    current = normalize_units_str(old_meta.get("units"))
    if is_dimensionless(current):
        raise ValueError(
            "Cannot convert from dimensionless units; use synapses.set_units() "
            "to assign spatial units first."
        )
    pending, target = _pending_units_metadata(
        old_meta,
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
        _scale_synapse_geometry(syn, factor)

    _commit_synapse_units(syn, pending)
    return syn


def snap_voxel_coordinates(
    syn: Synapses,
    *,
    method: _VoxelSnapMethod = "floor",
) -> Synapses:
    """Snap raw (+ mapped nearest) coordinates to integer voxel grid indices."""
    if not is_voxel_units(get_units(syn)):
        raise ValueError(
            "snap_voxel_coordinates() requires voxel units; convert to voxels before snapping."
        )
    validate_voxel_metadata(_synapse_meta_view(syn))

    if method not in ("floor", "round", "ceil"):
        raise ValueError("method must be 'floor', 'round', or 'ceil'.")

    def _snap(arr: np.ndarray) -> np.ndarray:
        if method == "floor":
            return np.floor(arr)
        if method == "round":
            return np.round(arr)
        return np.ceil(arr)

    syn._xyz = np.ascontiguousarray(_snap(np.asarray(syn._xyz, dtype=np.float64)))
    if syn.is_mapped and syn._nearest is not None:
        syn._nearest = np.ascontiguousarray(_snap(np.asarray(syn._nearest, dtype=np.float64)))
    return syn


def check_units_defined(syn: Synapses) -> None:
    """Raise if the synapse table has unset / dimensionless units or bad voxels."""
    if synapse_units_unset(syn) or is_dimensionless(get_units(syn)):
        raise ValueError(
            "Synapse units are unset or dimensionless; assign spatial units before converting."
        )
    if is_voxel_units(get_units(syn)):
        validate_voxel_metadata(_synapse_meta_view(syn))


def check_synapse_tree_units(
    syn: Synapses,
    tree: _Tree,
    *,
    context: str = "operation",
) -> None:
    """Warn on unset units; raise when both sides are set and disagree.

    Parameters
    ----------
    syn, tree
        Synapse table and morphology to compare.
    context
        Short label for warning/error messages (e.g. ``\"set_synapses\"``).
    """
    syn_unset = synapse_units_unset(syn)
    tree_unset = tree_units_unset(tree)

    if syn_unset and tree_unset:
        warnings.warn(
            f"{context}: synapse table and tree {tree.ID!r} both lack spatial "
            f"units; assign units before relying on mapped distances.",
            UserWarning,
            stacklevel=2,
        )
        return

    if syn_unset:
        warnings.warn(
            f"{context}: synapse table has no units; tree {tree.ID!r} uses "
            f"{get_tree_units(tree)!r}. Synapse coordinates will be treated as "
            f"matching the tree.",
            UserWarning,
            stacklevel=2,
        )
        return

    if tree_unset:
        warnings.warn(
            f"{context}: tree {tree.ID!r} has dimensionless units while "
            f"synapses declare {syn.units!r}. Assign tree units before mapping.",
            UserWarning,
            stacklevel=2,
        )
        return

    # Both declared — hard mismatch is an error.
    if not units_are_equal(
        syn.units,
        get_tree_units(tree),
        _synapse_units_meta(syn),
        _tree_units_meta(tree),
    ):
        raise ValueError(
            f"{context}: synapse units {syn.units!r} are incompatible with "
            f"tree {tree.ID!r} units {get_tree_units(tree)!r}. Convert one side "
            f"before binding or mapping."
        )


def stamp_synapse_units_from_tree(syn: Synapses, tree: _Tree) -> None:
    """Copy the tree's canonical units onto *syn* (including voxel metadata)."""
    units = get_tree_units(tree)
    syn.units = units
    if is_voxel_units(units):
        spec = get_tree_voxel_spec(tree)
        if spec is None:
            syn.units_meta = {}
        else:
            size, unit = spec
            syn.units_meta = {VOXEL_SIZE_KEY: size, VOXEL_UNIT_KEY: unit}
    else:
        syn.units_meta = {}


def apply_synapse_import_units(
    syn: Synapses,
    units: str | None,
    *,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Declare spatial units on an imported / extracted synapse table.

    Does not rescale coordinates (same contract as SWC / mesh ``set_units``).
    """
    if units is None:
        if voxel_size is not None or voxel_unit is not None:
            raise ValueError("voxel_size / voxel_unit require set_units")
        return
    set_units(syn, units, convert=False, voxel_size=voxel_size, voxel_unit=voxel_unit)


def sync_attached_synapse_units(tree: _Tree) -> None:
    """After a tree unit change, stamp attached synapses to match (if any)."""
    from ..tree_graphs.tree_synapses import _bind_synapses_gp, get_synapses

    syn = get_synapses(tree)
    if syn is None:
        return
    stamp_synapse_units_from_tree(syn, tree)
    _bind_synapses_gp(tree, syn)


__all__ = [
    "synapse_units_unset",
    "tree_units_unset",
    "get_units",
    "get_voxel_spec",
    "set_units",
    "set_voxel_units",
    "convert_units",
    "snap_voxel_coordinates",
    "check_units_defined",
    "check_synapse_tree_units",
    "stamp_synapse_units_from_tree",
    "apply_synapse_import_units",
    "sync_attached_synapse_units",
]
