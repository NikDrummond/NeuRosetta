"""Tests for Stone / Tree core identity, gp truth, and slots."""

import pytest

from neurosetta.api import Forest, Tree
from neurosetta.core import _Stone
from neurosetta.core.metadata import unwrap_metadata, wrap_metadata
from neurosetta.ops.tree_graphs import update_reduced


def test_tree_gp_is_source_of_truth(simple_tree):
    meta = {"units": "nm", "isReduced": False}
    tree = Tree(ID=1, metadata=meta, graph=simple_tree)

    assert "ID" in tree.graph.gp
    assert "metadata" in tree.graph.gp
    assert int(tree.graph.gp["ID"]) == 1
    assert tree.metadata is tree.graph.gp["metadata"]
    assert meta == {"units": "nm", "isReduced": False}
    assert tree.metadata == {**meta, "Flag": False}


def test_tree_id_setter_writes_gp(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    tree.ID = 99
    assert tree.ID == 99
    assert int(tree.graph.gp["ID"]) == 99


def test_tree_metadata_mutation_hits_gp(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    tree.metadata["Neuron_type"] = "T4"
    assert tree.graph.gp["metadata"]["Neuron_type"] == "T4"

    tree.metadata = {"units": "um"}
    assert tree.metadata is tree.graph.gp["metadata"]
    assert tree.metadata["units"] == "um"


def test_tree_slots_reject_arbitrary_attrs(simple_tree):
    tree = Tree(ID=1, metadata={}, graph=simple_tree)
    with pytest.raises(AttributeError):
        tree.foo = 1


def test_stone_slots_reject_arbitrary_attrs():
    s = _Stone(ID=1, metadata={})
    with pytest.raises(AttributeError):
        s.extra = True


def test_stone_copy_and_eq():
    stone = _Stone(ID=1, metadata={"key": "value"})
    copied = stone.copy()

    assert copied is not stone
    assert copied.ID == stone.ID
    assert copied.metadata == stone.metadata
    assert copied.metadata is not stone.metadata
    assert stone == copied

    copied.metadata["key"] = "other"
    assert stone.metadata["key"] == "value"
    assert stone != copied


def test_stone_repr():
    assert repr(_Stone(ID=42, metadata={})) == "_Stone(ID=42)"


def test_tree_eq_uses_graph_identity(simple_tree):
    tree_a = Tree(ID=1, metadata={}, graph=simple_tree)
    tree_b = Tree(ID=1, metadata={}, graph=simple_tree.copy())
    tree_c = tree_a.copy()

    assert tree_a == tree_a
    assert tree_a == Tree(ID=1, metadata={}, graph=simple_tree)
    assert tree_a != tree_b
    assert tree_a != tree_c
    assert tree_c.graph is not tree_a.graph


def test_from_graph_and_copy(simple_tree):
    tree = Tree(ID=5, metadata={"units": "nm"}, graph=simple_tree)
    wrapped = Tree.from_graph(tree.graph)
    assert wrapped.ID == 5
    assert wrapped.metadata is tree.metadata

    clone = tree.copy()
    assert clone.ID == tree.ID
    assert clone.graph is not tree.graph
    assert clone.metadata is not tree.metadata
    assert clone.metadata == tree.metadata
    clone.metadata["x"] = 1
    assert "x" not in tree.metadata


def test_update_reduced_via_metadata_property(simple_tree):
    tree = Tree(ID=1, metadata={"isReduced": False}, graph=simple_tree)
    update_reduced(tree)
    assert "isReduced" in tree.metadata
    assert tree.metadata["isReduced"] is tree.graph.gp["metadata"]["isReduced"]


def test_graph_swap_keeps_gp_truth(simple_tree):
    tree = Tree(ID=3, metadata={"units": "nm", "isReduced": False}, graph=simple_tree)
    new_g = tree.graph.copy()
    new_g.gp["metadata"] = wrap_metadata(dict(unwrap_metadata(new_g.gp["metadata"])))
    tree.graph = new_g
    assert tree.ID == 3
    assert tree.metadata is tree.graph.gp["metadata"]
    assert int(tree.graph.gp["ID"]) == 3
    update_reduced(tree)
    assert new_g.gp["metadata"]["isReduced"] is tree.metadata["isReduced"]


def test_forest_slots_and_ids(simple_tree):
    t1 = Tree(ID=1, metadata={}, graph=simple_tree)
    t2 = t1.copy()
    t2.ID = 2
    f = Forest([t1, t2])
    assert f.ids() == [1, 2]
    with pytest.raises(AttributeError):
        f.foo = 1


def test_forest_getitem_slice_and_list(simple_tree):
    base = Tree(ID=1, metadata={}, graph=simple_tree)
    trees = [base]
    for i in range(2, 6):
        tree = base.copy()
        tree.ID = i
        trees.append(tree)
    forest = Forest(trees)

    assert forest[2].ID == 3
    assert forest[1:4].ids() == [2, 3, 4]
    assert forest[[0, 2, 4]].ids() == [1, 3, 5]
    assert forest[(4, 1)].ids() == [5, 2]


def test_forest_getitem_boolean_mask(simple_tree):
    base = Tree(ID=1, metadata={}, graph=simple_tree)
    trees = [base]
    for i in range(2, 6):
        tree = base.copy()
        tree.ID = i
        trees.append(tree)
    forest = Forest(trees)

    assert forest[[True, False, True, False, True]].ids() == [1, 3, 5]

    import numpy as np

    mask = np.array([False, True, True, False, False])
    assert forest[mask].ids() == [2, 3]

    with pytest.raises(IndexError, match="boolean index did not match forest length"):
        forest[[True, False]]


def test_forest_by_id_single_and_list(simple_tree):
    base = Tree(ID=1, metadata={}, graph=simple_tree)
    trees = [base]
    for i in range(2, 4):
        tree = base.copy()
        tree.ID = i
        trees.append(tree)
    forest = Forest(trees)

    assert forest.by_id(2).ID == 2
    assert forest.by_id([3, 1]).ids() == [3, 1]
    assert forest.by_id((2, 2)).ids() == [2, 2]

    with pytest.raises(KeyError, match="No tree with ID 99"):
        forest.by_id(99)

    with pytest.raises(KeyError, match="No tree with ID 99"):
        forest.by_id([1, 99])
