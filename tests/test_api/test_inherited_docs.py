"""Ensure public API classes expose core helper docstrings via inheritance."""

from __future__ import annotations

import pytest

from neurosetta.api import Forest, Tree

TREE_CORE_METHODS = (
    "copy",
    "clone",
    "from_graph",
    "get_meta",
    "set_meta",
    "del_meta",
    "has_meta",
    "list_meta",
    "meta_summary",
    "set_flag",
    "list_properties",
    "has_property",
    "get_property",
    "set_property",
    "del_property",
    "revert_core_properties",
    "make_plot3d",
)

FOREST_CORE_METHODS = (
    "apply",
    "filter",
    "map",
    "build_3d",
    "append",
    "extend",
    "insert",
    "remove",
    "remove_id",
    "pop",
    "clear",
    "by_id",
    "ids",
    "copy",
    "get_meta",
    "set_meta",
    "del_meta",
    "has_meta",
    "list_meta",
    "meta_summary",
    "meta_indices",
    "list_properties",
)


@pytest.mark.parametrize("name", TREE_CORE_METHODS)
def test_tree_core_methods_have_docstrings(name: str):
    obj = getattr(Tree, name)
    assert obj.__doc__ and obj.__doc__.strip(), f"Tree.{name} missing docstring"


@pytest.mark.parametrize("name", FOREST_CORE_METHODS)
def test_forest_core_methods_have_docstrings(name: str):
    obj = getattr(Forest, name)
    assert obj.__doc__ and obj.__doc__.strip(), f"Forest.{name} missing docstring"


def test_forest_contains_has_docstring():
    assert Forest.__contains__.__doc__ and Forest.__contains__.__doc__.strip()
    assert "tree ID" in Forest.__contains__.__doc__


def test_tree_plot3d_property_has_docstring():
    assert Tree.plot3d.__doc__ and Tree.plot3d.__doc__.strip()
