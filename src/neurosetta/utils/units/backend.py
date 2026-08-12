"""Pint unit registry and neuroscience unit aliases."""

from __future__ import annotations

from pint import UnitRegistry

ureg = UnitRegistry()
ureg.define("micron = micrometer = um = µm")

DEFAULT_UNITS = "dimensionless"
TARGET_MICROMETER = "micron"
VOXEL_UNITS = "voxel"

LEGACY_UNDEFINED = frozenset({"undefined", "Undefined", ""})

__all__ = [
    "ureg",
    "DEFAULT_UNITS",
    "TARGET_MICROMETER",
    "VOXEL_UNITS",
    "LEGACY_UNDEFINED",
]
