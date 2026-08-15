"""Tests for Tree property list/get/set (scalar + vector<>)."""

import pytest
from graph_tool.all import Graph
from numpy import allclose, arange, array, stack

from neurosetta.api import Tree


def _node_coordinates(tree: Tree):
    g = tree.graph
    return stack([g.vp["x"].a, g.vp["y"].a, g.vp["z"].a], axis=1)


def test_list_properties(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    props = tree.list_properties()
    assert set(props) == {"g", "v", "e"}
    assert "ID" in props["g"]
    assert "metadata" in props["g"]
    assert {"x", "y", "z"}.issubset(props["v"])
    assert "radius" in props["v"]
    assert tree.list_properties("v") == props["v"]


def test_has_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    assert tree.has_property("x", "v")
    assert tree.has_property("ID", "g")
    assert tree.has_property("Path_length", "e")


def test_get_set_scalar_vertex_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    # level-agnostic
    rad = tree.get_property("radius")
    assert rad.shape == (17,)

    new = arange(17, dtype=float)
    tree.set_property("radius", new, "v")
    assert allclose(tree.get_property("radius"), new)


def test_get_set_xyz_coordinates(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)

    coords = _node_coordinates(tree)
    assert coords.shape == (17, 3)

    shifted = coords + 1.0
    tree.set_property("x", shifted[:, 0], "v")
    tree.set_property("y", shifted[:, 1], "v")
    tree.set_property("z", shifted[:, 2], "v")
    assert allclose(_node_coordinates(tree), shifted)


def test_set_scalar_on_vector_raises(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="is scalar"):
        tree.set_property("radius", array([[1.0, 2.0], [3.0, 4.0]]), "v")


def test_set_square_vector_warns_assume_soa():
    """Square (n, n): warn and assume SoA (d, n) for set_2d_array."""
    from numpy import eye

    g = Graph()
    g.add_vertex(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    # identity as SoA: dim i is one-hot on vertex i
    soa = eye(3)
    tree = Tree(ID=1, metadata={}, graph=g)
    tree.set_property("extra_vec", soa, "v", dtype="vector<double>", create=True)

    # overwrite with same square shape → warning, assume SoA
    with pytest.warns(UserWarning, match="assuming SoA"):
        tree.set_property("extra_vec", soa, "v")

    got_soa = tree.get_property("extra_vec", SoA=True)
    assert allclose(got_soa, soa)


def test_get_property_missing(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(KeyError, match="not found at any level"):
        tree.get_property("nope")
    with pytest.raises(KeyError, match="not found at level"):
        tree.get_property("nope", "v")


def test_get_property_name_collision_warns(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    n = tree.graph.num_vertices()
    m = tree.graph.num_edges()
    tree.set_property("flag", arange(n), "v", dtype="int", create=True)
    tree.set_property("flag", arange(m), "e", dtype="int", create=True)

    with pytest.warns(UserWarning, match="multiple levels"):
        result = tree.get_property("flag")

    assert set(result) == {"v", "e"}
    assert result["v"].shape == (n,)
    assert result["e"].shape == (m,)

    # explicit level still unambiguous
    assert tree.get_property("flag", "v").shape == (n,)


def test_create_and_delete_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    depths = arange(17, dtype=int)

    with pytest.raises(KeyError):
        tree.set_property("depth", depths, "v")

    tree.set_property("depth", depths, "v", dtype="int", create=True)
    assert tree.has_property("depth", "v")
    assert allclose(tree.get_property("depth"), depths)

    tree.del_property("depth", "v")
    assert not tree.has_property("depth", "v")


def test_create_vector_property(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    n = tree.graph.num_vertices()
    data = array([[i, i + 1] for i in range(n)], dtype=float)

    tree.set_property("extra_vec", data, "v", dtype="vector<double>", create=True)
    got = tree.get_property("extra_vec")
    assert got.shape == (n, 2)
    assert allclose(got, data)
    tree.del_property("extra_vec", "v")


def test_get_set_graph_property(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    assert tree.get_property("ID") == 1
    tree.set_property("ID", 42, "g")
    assert tree.ID == 42

    meta = tree.get_property("metadata")
    assert meta["units"] == "nm"
