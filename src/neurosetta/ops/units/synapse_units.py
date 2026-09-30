"""Unit checks and stamping for synapse tables bound to trees."""

from __future__ import annotations

import warnings
from typing import Any

from ...core import _Tree
from ...core.synapses import Synapses
from ...utils.units import (
    VOXEL_SIZE_KEY,
    VOXEL_UNIT_KEY,
    is_dimensionless,
    is_voxel_units,
    units_are_equal,
)
from .tree_units import get_units, get_voxel_spec


def synapse_units_unset(syn: Synapses) -> bool:
    """True when the synapse table has no declared spatial units."""
    return syn.units is None


def tree_units_unset(tree: _Tree) -> bool:
    """True when the tree has dimensionless / undeclared spatial units."""
    return is_dimensionless(get_units(tree))


def _synapse_units_meta(syn: Synapses) -> dict[str, Any]:
    return dict(syn.units_meta)


def _tree_units_meta(tree: _Tree) -> dict[str, Any]:
    return dict(tree.metadata)


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
            f"{get_units(tree)!r}. Synapse coordinates will be treated as "
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
        get_units(tree),
        _synapse_units_meta(syn),
        _tree_units_meta(tree),
    ):
        raise ValueError(
            f"{context}: synapse units {syn.units!r} are incompatible with "
            f"tree {tree.ID!r} units {get_units(tree)!r}. Convert one side "
            f"before binding or mapping."
        )


def stamp_synapse_units_from_tree(syn: Synapses, tree: _Tree) -> None:
    """Copy the tree's canonical units onto *syn* (including voxel metadata)."""
    units = get_units(tree)
    syn.units = units
    if is_voxel_units(units):
        spec = get_voxel_spec(tree)
        if spec is None:
            syn.units_meta = {}
        else:
            size, unit = spec
            syn.units_meta = {VOXEL_SIZE_KEY: size, VOXEL_UNIT_KEY: unit}
    else:
        syn.units_meta = {}


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
    "check_synapse_tree_units",
    "stamp_synapse_units_from_tree",
    "sync_attached_synapse_units",
]
