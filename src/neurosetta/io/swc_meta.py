"""Parse and format SWC comment-header metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path
from warnings import warn

from ..core import _Tree
from ..ops.units import get_units, get_voxel_spec
from ..utils.units import is_dimensionless, is_voxel_units

_META_RE = re.compile(r"^#\s*Meta\s*:\s*(.+)$", re.IGNORECASE)
_UNITS_RE = re.compile(r"^#\s*units\s*:\s*(.+)$", re.IGNORECASE)


def parse_swc_header(file_path: str | Path) -> dict:
    """Return metadata parsed from SWC comment lines."""
    meta: dict = {}
    with Path(file_path).open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break

            meta_match = _META_RE.match(stripped)
            if meta_match:
                try:
                    payload = json.loads(meta_match.group(1))
                except json.JSONDecodeError as exc:
                    warn(
                        f"Could not parse SWC Meta JSON in {file_path}: {exc}",
                        stacklevel=2,
                    )
                else:
                    if isinstance(payload, dict):
                        meta.update(payload)
                    else:
                        warn(
                            f"SWC Meta JSON in {file_path} must be an object; "
                            f"got {type(payload)!r}",
                            stacklevel=2,
                        )
                continue

            units_match = _UNITS_RE.match(stripped)
            if units_match:
                meta.setdefault("units", units_match.group(1).strip())

    return meta


def units_from_swc_header(header_meta: dict) -> str | None:
    """Extract a units string from parsed SWC header metadata."""
    units = header_meta.get("units")
    if units is None:
        return None
    return str(units)


def default_swc_header(tree: _Tree) -> str:
    """Build the default SWC header, including serialized metadata."""
    meta = {
        "id": str(tree.ID),
        "units": get_units(tree),
    }
    if (spec := get_voxel_spec(tree)) is not None:
        meta["voxel_size"], meta["voxel_unit"] = spec
    return f"SWC Generated using neurosetta\nMeta: {json.dumps(meta, sort_keys=True)}"


def swc_header_for_tree(tree: _Tree, header: str | None = None) -> str:
    """Return the SWC header text to pass to ``numpy.savetxt``."""
    if header is not None:
        return header
    return default_swc_header(tree)


def warn_if_export_dimensionless(tree: _Tree) -> None:
    """Warn when exporting a tree whose units are dimensionless."""
    if is_dimensionless(get_units(tree)):
        warn(
            f"Tree {tree.ID} has dimensionless units; exported SWC coordinates are "
            "not tagged with a spatial scale.",
            UserWarning,
            stacklevel=3,
        )
    elif is_voxel_units(get_units(tree)) and get_voxel_spec(tree) is None:
        warn(
            f"Tree {tree.ID} is tagged as voxel units but missing voxel metadata.",
            UserWarning,
            stacklevel=3,
        )


__all__ = [
    "parse_swc_header",
    "units_from_swc_header",
    "default_swc_header",
    "swc_header_for_tree",
    "warn_if_export_dimensionless",
]
