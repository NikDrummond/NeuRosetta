"""Tests for Stone / Tree core identity, gp truth, and slots."""

import pytest
from neurosetta.api import Tree, Forest
from neurosetta.core import _Stone
from neurosetta.ops.tree_graphs import update_reduced


def test_tree_gp_is_source_of_truth(simple_tree):
    meta = {"units": "nm", "isReduced": False}
    tree = Tree(ID=1, metadata=meta, graph=simple_tree)

    assert "ID" in tree.graph.gp
    assert "metadata" in tree.graph.gp
    assert int(tree.graph.gp["ID"]) == 1
    assert tree.metadata is tree.graph.gp["metadata"]
    assert tree.metadata is meta


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
    new_g.gp["metadata"] = dict(new_g.gp["metadata"])
    tree.graph = new_g
    assert tree.ID == 3
    assert tree.metadata is tree.graph.gp["metadata"]
    assert int(tree.graph.gp["ID"]) == 3
    tree.metadata["isReduced"] = True
    assert new_g.gp["metadata"]["isReduced"] is True


def test_forest_slots_and_ids(simple_tree):
    t1 = Tree(ID=1, metadata={}, graph=simple_tree)
    t2 = t1.copy()
    t2.ID = 2
    f = Forest([t1, t2])
    assert f.ids() == [1, 2]
    with pytest.raises(AttributeError):
        f.foo = 1
