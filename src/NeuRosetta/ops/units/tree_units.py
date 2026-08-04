"""Unit tracking and conversion for neuron trees."""

from __future__ import annotations

import warnings

from ...core import _Forest, _Tree
from ...utils.graph_utils import g_has_property, del_property
from ...utils.units import (
    DEFAULT_UNITS,
    TARGET_MICROMETER,
    VOXEL_UNITS,
    apply_voxel_metadata,
    clear_voxel_metadata,
    is_dimensionless,
    is_voxel_units,
    normalize_units_str,
    scale_factor,
    units_are_equal,
    validate_voxel_metadata,
    voxel_spec_from_metadata,
)


def _set_edge_lengths(tree: _Tree, factor: float) -> None:
    for prop in ("Path_length", "Euclidean_length"):
        if g_has_property(tree.graph, prop, "e"):

            tree.graph.ep[prop].a = tree.graph.ep[prop].a * factor
            # import here so not circular and re-calculate
            # from ...ops.tree_graphs.path_lengths import get_edge_length
            # # delete existing property and replace ( this could be smoother)
            # del_property(tree.graph, prop, 'e')
            # get_edge_length(tree, bind = True)


def _scale_tree_geometry(tree: _Tree, factor: float) -> None:
    coords = tree.get_node_coordinates() * factor
    tree.graph.vp['x'].a = coords[:,0]
    tree.graph.vp['y'].a = coords[:,1]
    tree.graph.vp['z'].a = coords[:,2]
    tree.graph.vp["radius"].a *= factor


def _pending_units_metadata(
    metadata: dict,
    units: str,
    *,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> tuple[dict, str]:
    pending = dict(metadata)
    target = normalize_units_str(units)
    if is_voxel_units(target):
        if voxel_size is None or voxel_unit is None:
            raise ValueError("Voxel units require voxel_size and voxel_unit.")
        apply_voxel_metadata(pending, voxel_size, voxel_unit)
        pending["units"] = VOXEL_UNITS
    else:
        clear_voxel_metadata(pending)
        pending["units"] = target
    return pending, target


def _commit_units_metadata(tree: _Tree, pending: dict) -> None:
    tree.metadata["units"] = pending["units"]
    if is_voxel_units(pending["units"]):
        tree.metadata["voxel_size"] = pending["voxel_size"]
        tree.metadata["voxel_unit"] = pending["voxel_unit"]
    else:
        clear_voxel_metadata(tree.metadata)


def get_units(tree: _Tree) -> str:
    """Return canonical spatial units for a tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    str
        Canonical unit string.
    """
    return normalize_units_str(tree.metadata.get("units"))


def get_voxel_spec(tree: _Tree) -> tuple[float, str] | None:
    """Return voxel metadata when the tree uses voxel coordinates.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    tuple[float, str] | None
        ``(voxel_size, voxel_unit)`` when the tree uses voxel coordinates;
        otherwise None.
    """
    return voxel_spec_from_metadata(tree.metadata)


def set_units(
    tree: _Tree,
    units: str,
    *,
    convert: bool = False,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> None:
    """Set spatial units in tree metadata, optionally rescaling geometry.

    Parameters
    ----------
    tree : _Tree
        Target tree.
    units : str
        Canonical unit string or ``"voxel"``.
    convert : bool, optional
        Rescale coordinates, radii, and edge lengths when changing units.
        By default False.
    voxel_size : float | None, optional
        Cubic edge length when *units* is voxel-based. By default None.
    voxel_unit : str | None, optional
        Physical unit of each voxel edge when *units* is voxel-based.
        By default None.

    Returns
    -------
    None
    """
    old_meta = dict(tree.metadata)
    current = normalize_units_str(old_meta.get("units"))
    pending, target = _pending_units_metadata(
        tree.metadata,
        units,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )

    if convert and not units_are_equal(current, target, old_meta, pending):
        if not is_dimensionless(current):
            factor = scale_factor(
                current,
                target,
                from_metadata=old_meta,
                to_metadata=pending,
            )
            _scale_tree_geometry(tree, factor)
            _set_edge_lengths(tree, factor)

    _commit_units_metadata(tree, pending)


def set_voxel_units(
    tree: _Tree,
    voxel_size: float,
    voxel_unit: str,
) -> None:
    """Tag coordinates as voxel indices with a cubic edge length.

    Parameters
    ----------
    tree : _Tree
        Target tree.
    voxel_size : float
        Physical length of one cubic voxel edge.
    voxel_unit : str
        Physical unit of *voxel_size*.

    Returns
    -------
    None
    """
    set_units(
        tree,
        VOXEL_UNITS,
        voxel_size=voxel_size,
        voxel_unit=voxel_unit,
    )


def convert_units(
    tree: _Tree,
    target_units: str,
    *,
    in_place: bool = True,
    voxel_size: float | None = None,
    voxel_unit: str | None = None,
) -> _Tree:
    """Convert tree coordinates and radii to target units.

    Parameters
    ----------
    tree : _Tree
        Tree to convert.
    target_units : str
        Destination unit string.
    in_place : bool, optional
        Mutate *tree* in place. By default True.
    voxel_size : float | None, optional
        Required when converting to or from voxel units. By default None.
    voxel_unit : str | None, optional
        Required when converting to or from voxel units. By default None.

    Returns
    -------
    _Tree
        Converted tree (same object when in_place=True).
    """
    if not in_place:
        tree = tree.copy()

    old_meta = dict(tree.metadata)
    current = normalize_units_str(old_meta.get("units"))
    pending, target = _pending_units_metadata(
        tree.metadata,
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
        _scale_tree_geometry(tree, factor)
        _set_edge_lengths(tree, factor)

    _commit_units_metadata(tree, pending)
    return tree


def check_units_defined(tree: _Tree) -> None:
    """Verify that a tree has non-dimensionless spatial units.

    Parameters
    ----------
    tree : _Tree
        Tree to check.

    Raises
    ------
    ValueError
        If units are dimensionless or voxel metadata is invalid.
    """
    if is_dimensionless(get_units(tree)):
        raise ValueError(
            "Tree units are dimensionless; assign spatial units before converting."
        )
    if is_voxel_units(get_units(tree)):
        validate_voxel_metadata(tree.metadata)


def _defined_unit_spec(tree: _Tree) -> tuple[str, tuple[float, str] | None]:
    units = get_units(tree)
    if is_voxel_units(units):
        return units, validate_voxel_metadata(tree.metadata)
    return units, None


def harmonize_forest_units(
    forest: _Forest,
    target_units: str = TARGET_MICROMETER,
) -> None:
    """Convert all trees in a forest to a common target unit.

    Emits a warning when trees carry different spatial units and skips trees
    that remain dimensionless.

    Parameters
    ----------
    forest : _Forest
        Forest to harmonize.
    target_units : str, optional
        Target unit for conversion. By default micrometers.

    Returns
    -------
    None
    """
    target = normalize_units_str(target_units)
    defined = [
        _defined_unit_spec(tree)
        for tree in forest
        if not is_dimensionless(get_units(tree))
    ]

    if len(set(defined)) > 1:
        warnings.warn(
            f"Forest contains mixed units {[str(spec) for spec in set(defined)]}; "
            f"converting defined trees to {target!r}.",
            UserWarning,
            stacklevel=2,
        )

    for tree in forest:
        current = get_units(tree)
        if is_dimensionless(current):
            warnings.warn(
                f"Tree {tree.ID} has dimensionless units; skipping conversion.",
                UserWarning,
                stacklevel=2,
            )
            continue
        convert_units(tree, target, in_place=True)


def ensure_forest_units(
    forest: _Forest,
    target_units: str = TARGET_MICROMETER,
) -> None:
    """Harmonize forest units and require spatial units on every tree.

    Parameters
    ----------
    forest : _Forest
        Forest to validate.
    target_units : str, optional
        Target unit for harmonization. By default micrometers.

    Raises
    ------
    ValueError
        If any tree remains dimensionless after harmonization.
    """
    harmonize_forest_units(forest, target_units=target_units)
    dimensionless_ids = [
        tree.ID for tree in forest if is_dimensionless(get_units(tree))
    ]
    if dimensionless_ids:
        raise ValueError(
            "Forest contains trees with dimensionless units: "
            f"{dimensionless_ids}. Assign units before running spatial forest operations."
        )


__all__ = [
    "DEFAULT_UNITS",
    "get_units",
    "get_voxel_spec",
    "set_units",
    "set_voxel_units",
    "convert_units",
    "check_units_defined",
    "harmonize_forest_units",
    "ensure_forest_units",
]
