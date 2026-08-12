"""Documented unit aliases and reference tables."""

from __future__ import annotations

from dataclasses import dataclass

from .backend import VOXEL_UNITS


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    """One canonical spatial unit and its accepted aliases."""

    canonical: str
    aliases: tuple[str, ...]
    notes: str = ""


UNIT_DEFINITIONS: tuple[UnitDefinition, ...] = (
    UnitDefinition(
        canonical="dimensionless",
        aliases=("dimensionless", "undefined", "Undefined", "1 dimensionless"),
        notes="Default when no spatial scale is assigned.",
    ),
    UnitDefinition(
        canonical="nanometer",
        aliases=("nm", "nanometer", "Nanometer"),
        notes="Common for EM reconstructions.",
    ),
    UnitDefinition(
        canonical="micron",
        aliases=("micron", "micrometer", "Micron","Micrometer","Microns","um", "µm", "microns"),
        notes="Default harmonization target for forests.",
    ),
    UnitDefinition(
        canonical="millimeter",
        aliases=("mm", "millimeter"),
    ),
    UnitDefinition(
        canonical="meter",
        aliases=("m", "meter"),
    ),
    UnitDefinition(
        canonical=VOXEL_UNITS,
        aliases=("voxel",),
        notes=(
            "Voxel-index coordinates. Requires ``metadata['voxel_size']`` and "
            "``metadata['voxel_unit']`` (edge length per voxel)."
        ),
    ),
)


def list_unit_definitions() -> tuple[UnitDefinition, ...]:
    """Return the documented unit definitions."""
    return UNIT_DEFINITIONS


def alias_lookup() -> dict[str, str]:
    """Map every accepted alias to its canonical unit string."""
    lookup: dict[str, str] = {}
    for definition in UNIT_DEFINITIONS:
        for alias in definition.aliases:
            lookup[alias] = definition.canonical
    return lookup


def format_units_reference_table() -> str:
    """Return a markdown table of canonical units and aliases."""
    lines = [
        "| Canonical | Aliases | Notes |",
        "| --- | --- | --- |",
    ]
    for definition in UNIT_DEFINITIONS:
        aliases = ", ".join(f"``{alias}``" for alias in definition.aliases)
        notes = definition.notes.replace("|", "\\|")
        lines.append(f"| ``{definition.canonical}`` | {aliases} | {notes} |")
    return "\n".join(lines)


__all__ = [
    "UnitDefinition",
    "UNIT_DEFINITIONS",
    "list_unit_definitions",
    "alias_lookup",
    "format_units_reference_table",
]
