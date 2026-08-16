"""Tests for Tree user-metadata CRUD with protected core keys."""

import pytest

from neurosetta.api import Tree
from neurosetta.core.metadata import PROTECTED_META_KEYS

PROTECTED = sorted(PROTECTED_META_KEYS)


def test_set_get_del_meta(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    tree.set_meta("Neuron_type", "T4")
    assert tree.get_meta("Neuron_type") == "T4"
    assert tree.has_meta("Neuron_type")
    assert "Neuron_type" in tree.list_meta()
    tree.del_meta("Neuron_type")
    assert not tree.has_meta("Neuron_type")
    assert tree.get_meta("missing", default=42) == 42


@pytest.mark.parametrize("key", PROTECTED)
def test_protected_meta_set_raises(simple_tree, key):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    with pytest.raises(KeyError, match="protected"):
        tree.set_meta(key, "x")


@pytest.mark.parametrize("key", PROTECTED)
def test_protected_meta_del_raises(simple_tree, key):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    with pytest.raises(KeyError, match="protected"):
        tree.del_meta(key)


def test_list_meta_excludes_protected(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm", "file_path": "/a.swc"}, graph=simple_tree)
    tree.set_meta("score", 1.0)
    assert tree.list_meta() == ["score"]
    all_keys = set(tree.list_meta(include_protected=True))
    assert {"units", "file_path", "score"}.issubset(all_keys)


def test_meta_summary(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    tree.set_meta("Neuron_type", "T4")
    tree.set_meta("score", 1.0)

    assert tree.meta_summary() == {"Neuron_type": 1, "score": 1}
    assert "units" in tree.meta_summary(include_protected=True)


def test_direct_protected_metadata_mutation_raises(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    with pytest.raises(KeyError, match="protected"):
        tree.metadata["units"] = "um"


def test_metadata_bulk_replace_via_setter(simple_tree):
    tree = Tree(ID=1, metadata={"units": "nm"}, graph=simple_tree)
    tree.metadata = {"units": "um", "Neuron_type": "T4"}
    assert tree.metadata["units"] == "um"
    assert tree.metadata["Neuron_type"] == "T4"
