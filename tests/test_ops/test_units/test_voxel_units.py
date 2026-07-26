"""Tests for voxel unit operations on trees."""

import pytest
from numpy import array_equal

from NeuRosetta.api import Tree
from NeuRosetta.io.swc_utils import export_swc, import_swc
from NeuRosetta.ops.units import convert_units, get_units, get_voxel_spec, set_voxel_units


def test_set_voxel_units_metadata(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    set_voxel_units(tree, 8, "nm")

    assert get_units(tree) == "voxel"
    assert get_voxel_spec(tree) == (8.0, "nanometer")


def test_convert_voxel_units_to_micron(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    set_voxel_units(tree, 8, "nm")
    coords_before = tree.graph.vp["coordinates"].get_2d_array().copy()

    convert_units(tree, "micron")

    assert get_units(tree) == "micron"
    assert get_voxel_spec(tree) is None
    assert array_equal(
        tree.graph.vp["coordinates"].get_2d_array(),
        coords_before * 0.008,
    )


def test_import_swc_voxel_units(tmp_path):
    swc = tmp_path / "1.swc"
    swc.write_text("1 1 0.0 0.0 0.0 0.5 -1\n")

    tree = import_swc(swc, set_units="voxel", voxel_size=8, voxel_unit="nm")

    assert get_units(tree) == "voxel"
    assert get_voxel_spec(tree) == (8.0, "nanometer")


def test_export_import_voxel_round_trip(tmp_path, simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    set_voxel_units(tree, 8, "nm")
    path = tmp_path / "1.swc"

    export_swc(tree, path)
    loaded = import_swc(path)

    assert get_units(loaded) == "voxel"
    assert get_voxel_spec(loaded) == (8.0, "nanometer")


def test_set_voxel_units_requires_pair(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="voxel_size and voxel_unit"):
        tree.set_units("voxel")
