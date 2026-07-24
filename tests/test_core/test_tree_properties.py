"""Tests for Tree property list/get/set (scalar + vector<>)."""

import pytest
from numpy import array, allclose, arange

from NeuRosetta.api import Tree


def test_list_properties(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    props = tree.list_properties()
    assert set(props) == {"g", "v", "e"}
    assert "ID" in props["g"]
    assert "metadata" in props["g"]
    assert "coordinates" in props["v"]
    assert "radius" in props["v"]
    assert tree.list_properties("v") == props["v"]


def test_has_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    assert tree.has_property("coordinates", "v")
    assert tree.has_property("ID", "g")
    assert not tree.has_property("Path_length", "e")


def test_get_set_scalar_vertex_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    rad = tree.get_property("radius", "v", as_array=True)
    assert rad.shape == (17,)

    new = arange(17, dtype=float)
    tree.set_property("radius", new, "v")
    assert allclose(tree.get_property("radius", "v", as_array=True), new)


def test_get_set_coordinates_vector(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)

    # default AoS (N, 3)
    coords = tree.get_property("coordinates", "v", as_array=True)
    assert coords.shape == (17, 3)

    soa = tree.get_property("coordinates", "v", as_array=True, SoA=True)
    assert soa.shape == (3, 17)
    assert allclose(soa.T, coords)

    # PropertyMap path (not array)
    prop = tree.get_property("coordinates", "v")
    assert prop.value_type() == "vector<double>"

    # set via AoS
    shifted = coords + 1.0
    tree.set_property("coordinates", shifted, "v")
    assert allclose(
        tree.get_property("coordinates", "v", as_array=True), shifted
    )

    # set via SoA
    tree.set_property("coordinates", coords.T, "v", SoA=True)
    assert allclose(tree.get_property("coordinates", "v", as_array=True), coords)


def test_create_and_delete_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    depths = arange(17, dtype=int)

    with pytest.raises(KeyError):
        tree.set_property("depth", depths, "v")

    tree.set_property("depth", depths, "v", dtype="int", create=True)
    assert tree.has_property("depth", "v")
    assert allclose(tree.get_property("depth", "v", as_array=True), depths)

    tree.del_property("depth", "v")
    assert not tree.has_property("depth", "v")


def test_create_vector_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    n = tree.graph.num_vertices()
    data = array([[i, i + 1] for i in range(n)], dtype=float)

    tree.set_property(
        "extra_vec", data, "v", dtype="vector<double>", create=True
    )
    got = tree.get_property("extra_vec", "v", as_array=True)
    assert got.shape == (n, 2)
    assert allclose(got, data)
    tree.del_property("extra_vec", "v")


def test_get_set_graph_property(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    assert tree.get_property("ID", "g", as_array=True) == 1
    tree.set_property("ID", 42, "g")
    assert tree.ID == 42

    meta = tree.get_property("metadata", "g", as_array=True)
    assert meta["units"] == "nm"
