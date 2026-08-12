"""Parse and normalize unit strings stored in tree metadata."""

from __future__ import annotations

from pint.util import UnitsContainer

from .backend import DEFAULT_UNITS, LEGACY_UNDEFINED, VOXEL_UNITS, ureg

Unit = UnitsContainer


def is_voxel_units(units: str | None) -> bool:
    """Return True when ``units`` denotes voxel-index coordinates."""
    raw = _coerce_units_str(units)
    return raw == VOXEL_UNITS


def _coerce_units_str(units: str | None) -> str:
    if units is None or units in LEGACY_UNDEFINED:
        return DEFAULT_UNITS
    return units


def normalize_units_str(units: str | None) -> str:
    """Return a canonical unit string suitable for ``metadata['units']``."""
    raw = _coerce_units_str(units)
    if raw == DEFAULT_UNITS:
        return DEFAULT_UNITS
    if is_voxel_units(raw):
        return VOXEL_UNITS
    return str(parse_units(raw))


def parse_units(units: str | None) -> Unit:
    """Parse a metadata units string into a Pint unit."""
    raw = _coerce_units_str(units)
    if is_voxel_units(raw):
        raise ValueError("Voxel units are metadata-backed; use voxel_size and voxel_unit instead.")
    try:
        return ureg.parse_units(raw)
    except Exception as exc:
        raise ValueError(f"Unknown units {units!r}") from exc


def is_dimensionless(units: str | None) -> bool:
    """Return True when units are undefined / dimensionless."""
    return normalize_units_str(units) == DEFAULT_UNITS


__all__ = [
    "Unit",
    "DEFAULT_UNITS",
    "normalize_units_str",
    "parse_units",
    "is_voxel_units",
    "is_dimensionless",
]
