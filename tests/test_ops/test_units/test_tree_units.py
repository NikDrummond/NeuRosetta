"""Tests for tree and forest unit operations."""

import warnings

import pytest
from numpy import array_equal

from neurosetta.api import Forest, Tree
from neurosetta.ops.units import (
    check_units_defined,
    convert_units,
    ensure_forest_units,
    get_units,
    harmonize_forest_units,
    set_units,
)
from neurosetta.utils.units import DEFAULT_UNITS


def test_get_units_defaults_to_dimensionless(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    assert get_units(tree) == DEFAULT_UNITS


def test_set_units_normalizes_metadata(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    set_units(tree, "um", convert=False)
    assert get_units(tree) == "micron"


def test_convert_units_scales_geometry(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    coords_before = tree.get_node_coordinates().copy()
    radius_before = tree.graph.vp["radius"].a.copy()

    convert_units(tree, "micron")

    assert get_units(tree) == "micron"
    assert array_equal(tree.get_node_coordinates(), coords_before * 0.001)
    assert array_equal(tree.graph.vp["radius"].a, radius_before * 0.001)


def test_convert_units_recreates_edge_lengths(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    tree.get_edge_length()
    lengths_before = tree.graph.ep["Path_length"].a.copy()
    assert "Path_length" in tree.graph.ep

    convert_units(tree, "micron")

    assert "Path_length" in tree.graph.ep
    assert array_equal(tree.graph.ep["Path_length"].a, lengths_before * 0.001)


def test_convert_units_in_place_false_returns_copy(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    original_units = get_units(tree)
    converted = convert_units(tree, "micron", in_place=False)

    assert converted is not tree
    assert get_units(tree) == original_units
    assert get_units(converted) == "micron"


def test_check_units_defined_raises(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="dimensionless"):
        check_units_defined(tree)


def test_harmonize_forest_units_warns_and_converts(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = Tree(ID=2, metadata={"units": "micron"}, graph=simple_tree.copy())
    forest = Forest([t1, t2])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        harmonize_forest_units(forest)

    assert any("mixed units" in str(w.message) for w in caught)
    assert get_units(t1) == "micron"
    assert get_units(t2) == "micron"


def test_harmonize_forest_skips_dimensionless(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = Tree(ID=2, metadata={}, graph=simple_tree.copy())
    forest = Forest([t1, t2])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        harmonize_forest_units(forest)

    assert any("dimensionless" in str(w.message) for w in caught)
    assert get_units(t1) == "micron"
    assert get_units(t2) == DEFAULT_UNITS


def test_ensure_forest_units_raises_on_dimensionless(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = Tree(ID=2, metadata={}, graph=simple_tree.copy())
    forest = Forest([t1, t2])

    with pytest.raises(ValueError, match="dimensionless"):
        ensure_forest_units(forest)


def test_ensure_forest_units_harmonizes_defined_trees(simple_tree):
    t1 = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    t2 = Tree(ID=2, metadata={"units": "micron"}, graph=simple_tree.copy())
    forest = Forest([t1, t2])

    ensure_forest_units(forest)

    assert get_units(t1) == "micron"
    assert get_units(t2) == "micron"
