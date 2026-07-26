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
    # level-agnostic
    rad = tree.get_property("radius")
    assert rad.shape == (17,)

    new = arange(17, dtype=float)
    tree.set_property("radius", new, "v")
    assert allclose(tree.get_property("radius"), new)


def test_get_set_coordinates_vector(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)

    # default get → AoS (N, 3)
    coords = tree.get_property("coordinates")
    assert coords.shape == (17, 3)

    soa = tree.get_property("coordinates", SoA=True)
    assert soa.shape == (3, 17)
    assert allclose(soa.T, coords)

    prop = tree.get_property("coordinates", "v", as_array=False)
    assert prop.value_type() == "vector<double>"

    # set AoS (n, d) — coerced to (d, n)
    shifted = coords + 1.0
    tree.set_property("coordinates", shifted, "v")
    assert allclose(tree.get_property("coordinates"), shifted)

    # set SoA (d, n) — used as-is
    tree.set_property("coordinates", coords.T, "v")
    assert allclose(tree.get_property("coordinates"), coords)


def test_set_vector_rejects_bad_shape(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="does not match level size"):
        tree.set_property("coordinates", array([[1.0, 2.0], [3.0, 4.0]]), "v")


def test_set_scalar_on_vector_raises(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    before = tree.get_property("coordinates").copy()
    with pytest.raises(ValueError, match="is vector<>"):
        tree.set_property("coordinates", arange(17, dtype=float), "v")
    assert allclose(tree.get_property("coordinates"), before)


def test_set_vector_on_scalar_raises(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(ValueError, match="is scalar"):
        tree.set_property("radius", array([[1.0, 2.0], [3.0, 4.0]]), "v")


def test_set_square_vector_warns_assume_soa():
    """Square (n, n): warn and assume SoA (d, n) for set_2d_array."""
    from graph_tool.all import Graph
    from numpy import eye

    g = Graph()
    g.add_vertex(3)
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    # identity as SoA: dim i is one-hot on vertex i
    soa = eye(3)
    tree = Tree(ID=1, metadata={}, graph=g)
    tree.set_property("coordinates", soa, "v", dtype="vector<double>", create=True)

    # overwrite with same square shape → warning, assume SoA
    with pytest.warns(UserWarning, match="assuming SoA"):
        tree.set_property("coordinates", soa, "v")

    got_soa = tree.get_property("coordinates", SoA=True)
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

    tree.set_property(
        "extra_vec", data, "v", dtype="vector<double>", create=True
    )
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
