"""Tests for documented unit aliases."""

from NeuRosetta.utils.units.aliases import alias_lookup, format_units_reference_table, list_unit_definitions


def test_alias_lookup_maps_nm_to_nanometer():
    assert alias_lookup()["nm"] == "nanometer"
    assert alias_lookup()["µm"] == "micron"


def test_list_unit_definitions_includes_voxel():
    canonical = {item.canonical for item in list_unit_definitions()}
    assert "voxel" in canonical
    assert "nanometer" in canonical


def test_format_units_reference_table_markdown():
    table = format_units_reference_table()
    assert "| Canonical | Aliases | Notes |" in table
    assert "``nanometer``" in table
    assert "``voxel``" in table
