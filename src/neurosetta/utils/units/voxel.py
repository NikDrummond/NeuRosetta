"""Voxel coordinate units backed by edge length metadata."""

from __future__ import annotations

from .backend import VOXEL_UNITS
from .parsing import normalize_units_str, parse_units

VOXEL_SIZE_KEY = "voxel_size"
VOXEL_UNIT_KEY = "voxel_unit"


def validate_voxel_metadata(metadata: dict) -> tuple[float, str]:
    """Return normalized ``(voxel_size, voxel_unit)`` from tree metadata."""
    if VOXEL_SIZE_KEY not in metadata or VOXEL_UNIT_KEY not in metadata:
        raise ValueError(
            "Voxel units require metadata keys "
            f"'{VOXEL_SIZE_KEY}' and '{VOXEL_UNIT_KEY}'."
        )

    size = float(metadata[VOXEL_SIZE_KEY])
    if size <= 0:
        raise ValueError(f"voxel_size must be positive; got {size!r}.")

    unit = normalize_units_str(str(metadata[VOXEL_UNIT_KEY]))
    if unit == "dimensionless":
        raise ValueError("voxel_unit must be a spatial unit, not dimensionless.")

    return size, unit


def apply_voxel_metadata(metadata: dict, voxel_size: float, voxel_unit: str) -> None:
    """Write normalized voxel metadata fields."""
    size, unit = validate_voxel_metadata(
        {VOXEL_SIZE_KEY: voxel_size, VOXEL_UNIT_KEY: voxel_unit}
    )
    metadata[VOXEL_SIZE_KEY] = size
    metadata[VOXEL_UNIT_KEY] = unit


def clear_voxel_metadata(metadata: dict) -> None:
    """Remove voxel-specific metadata keys."""
    metadata.pop(VOXEL_SIZE_KEY, None)
    metadata.pop(VOXEL_UNIT_KEY, None)


def meters_per_coordinate_unit(units: str | None, metadata: dict | None = None) -> float:
    """Return SI meters represented by one coordinate unit."""
    normalized = normalize_units_str(units)
    if normalized == "dimensionless":
        raise ValueError("Cannot compute scale for dimensionless units.")

    if normalized == VOXEL_UNITS:
        if metadata is None:
            raise ValueError("Voxel units require metadata for scale conversion.")
        size, base_unit = validate_voxel_metadata(metadata)
        base_meters = float((1 * parse_units(base_unit)).to_base_units().magnitude)
        return size * base_meters

    return float((1 * parse_units(normalized)).to_base_units().magnitude)


def voxel_spec_from_metadata(metadata: dict) -> tuple[float, str] | None:
    """Return ``(voxel_size, voxel_unit)`` when metadata describes voxels."""
    if normalize_units_str(metadata.get("units")) != VOXEL_UNITS:
        return None
    size, unit = validate_voxel_metadata(metadata)
    return size, unit


__all__ = [
    "VOXEL_SIZE_KEY",
    "VOXEL_UNIT_KEY",
    "validate_voxel_metadata",
    "apply_voxel_metadata",
    "clear_voxel_metadata",
    "meters_per_coordinate_unit",
    "voxel_spec_from_metadata",
]
