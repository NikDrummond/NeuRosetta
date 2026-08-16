"""Tests for voxel unit operations on trees."""

import numpy as np
import pytest
from numpy import array_equal

from neurosetta.api import Tree
from neurosetta.io.swc_utils import export_swc, import_swc
from neurosetta.ops.units import (
    convert_units,
    get_units,
    get_voxel_spec,
    set_voxel_units,
    snap_voxel_coordinates,
)


def test_set_voxel_units_metadata(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    set_voxel_units(tree, 8, "nm")

    assert get_units(tree) == "voxel"
    assert get_voxel_spec(tree) == (8.0, "nanometer")


def test_set_units_infers_voxel_from_size_and_unit(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    tree.set_units(voxel_size=8, voxel_unit="nm")

    assert get_units(tree) == "voxel"
    assert get_voxel_spec(tree) == (8.0, "nanometer")


def test_convert_voxel_units_to_micron(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    set_voxel_units(tree, 8, "nm")
    coords_before = tree.get_node_coordinates().copy()

    convert_units(tree, "micron")

    assert get_units(tree) == "micron"
    assert get_voxel_spec(tree) is None
    assert array_equal(tree.get_node_coordinates(), coords_before * 0.008)


def test_convert_units_infers_voxel_from_size_and_unit(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    coords_before = tree.get_node_coordinates().copy()
    radius_before = tree.graph.vp["radius"].a.copy()

    tree.convert_units(voxel_size=8, voxel_unit="nm")

    assert get_units(tree) == "voxel"
    assert get_voxel_spec(tree) == (8.0, "nanometer")
    assert array_equal(tree.get_node_coordinates(), coords_before / 8.0)
    assert array_equal(tree.graph.vp["radius"].a, radius_before / 8.0)


def test_convert_units_requires_target_or_voxel_pair(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    with pytest.raises(ValueError, match="target_units, or both voxel_size and voxel_unit"):
        convert_units(tree)


def test_convert_units_raises_from_dimensionless(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="dimensionless"):
        convert_units(tree, "nm")


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


def test_snap_voxel_coordinates_to_integer_indices(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    convert_units(tree, voxel_size=80, voxel_unit="nm")

    snapped = snap_voxel_coordinates(tree)
    coords = snapped.get_node_coordinates()

    assert np.allclose(coords, np.floor(coords))
    assert len(np.unique(coords, axis=0)) <= tree.graph.num_vertices()


def test_snap_voxel_coordinates_recalculates_edge_lengths(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    convert_units(tree, voxel_size=80, voxel_unit="nm")
    lengths_before = tree.graph.ep["Path_length"].a.copy()

    snap_voxel_coordinates(tree)

    assert not np.allclose(tree.graph.ep["Path_length"].a, lengths_before)


def test_snap_voxel_coordinates_collapses_example_neuron():
    import neurosetta as nr

    tree = nr.load_example_data(nr.example_ids[0])
    tree.convert_units(voxel_size=80, voxel_unit="um")
    coords_before = tree.get_node_coordinates()
    assert not np.allclose(coords_before, np.floor(coords_before))

    tree.snap_voxel_coordinates()

    coords = tree.get_node_coordinates()
    assert np.allclose(coords, np.floor(coords))
    assert len(np.unique(coords, axis=0)) == 3


def test_snap_voxel_coordinates_requires_voxel_units(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    with pytest.raises(ValueError, match="requires voxel units"):
        snap_voxel_coordinates(tree)


def test_snap_voxel_coordinates_forest(simple_tree):
    from neurosetta.api import Forest

    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = Tree(ID=2, metadata={"units": "nm"}, graph=simple_tree.copy())
    convert_units(t1, voxel_size=80, voxel_unit="nm")
    convert_units(t2, voxel_size=80, voxel_unit="nm")
    forest = Forest([t1, t2])

    forest.snap_voxel_coordinates()

    for tree in forest:
        coords = tree.get_node_coordinates()
        assert np.allclose(coords, np.floor(coords))
