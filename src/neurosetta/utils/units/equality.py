"""Compare normalized unit strings, including voxel metadata."""

from __future__ import annotations

from .parsing import is_voxel_units, normalize_units_str
from .voxel import validate_voxel_metadata


def units_are_equal(
    a: str | None,
    b: str | None,
    metadata_a: dict | None = None,
    metadata_b: dict | None = None,
) -> bool:
    """Return True when two metadata unit strings refer to the same quantity.

    Parameters
    ----------
    a : str or None
        First unit string.
    b : str or None
        Second unit string.
    metadata_a : dict or None, optional
        Metadata for ``a``. Required when comparing voxel units.
    metadata_b : dict or None, optional
        Metadata for ``b``. Required when comparing voxel units.

    Returns
    -------
    bool
        True when the normalized unit strings match and, for voxel units,
        the voxel metadata is identical.
    """
    norm_a = normalize_units_str(a)
    norm_b = normalize_units_str(b)
    if norm_a != norm_b:
        return False
    if is_voxel_units(norm_a):
        if metadata_a is None or metadata_b is None:
            return False
        return validate_voxel_metadata(metadata_a) == validate_voxel_metadata(metadata_b)
    return True


__all__ = ["units_are_equal"]
