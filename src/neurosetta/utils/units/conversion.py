"""Unit conversion factors for spatial geometry."""

from __future__ import annotations

from .equality import units_are_equal
from .parsing import is_dimensionless, normalize_units_str
from .voxel import meters_per_coordinate_unit


def scale_factor(
    from_units: str | None,
    to_units: str | None,
    *,
    from_metadata: dict | None = None,
    to_metadata: dict | None = None,
) -> float:
    """Return the multiplier applied to coordinates in ``from_units`` to reach ``to_units``."""
    from_norm = normalize_units_str(from_units)
    to_norm = normalize_units_str(to_units)

    if from_norm == to_norm:
        if from_metadata is None or to_metadata is None:
            return 1.0

        if units_are_equal(from_norm, to_norm, from_metadata, to_metadata):
            return 1.0

    if is_dimensionless(from_norm):
        raise ValueError(
            "Cannot convert from dimensionless units; set spatial units before converting."
        )
    if is_dimensionless(to_norm):
        raise ValueError("Cannot convert spatial coordinates to dimensionless units.")

    from_m = meters_per_coordinate_unit(from_norm, from_metadata)
    to_m = meters_per_coordinate_unit(to_norm, to_metadata)
    return from_m / to_m


__all__ = ["scale_factor"]
