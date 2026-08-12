"""Tests for graph_utils.gt_properties."""

import warnings

import numpy as np
import pytest
from graph_tool.all import Graph

from neurosetta.utils.graph_utils.gt_properties import (
    _InternalPropertyMissingError,
    bind_edge_property,
    bind_graph_property,
    bind_vertex_property,
    del_property,
    g_has_property,
    get_property,
    list_properties,
    raise_internal_property_missing,
    revert_core_properties,
    set_property,
)


def test_g_has_property(simple_tree):
    assert g_has_property(simple_tree, "x", "v")
    assert not g_has_property(simple_tree, "missing", "v")


def test_list_properties(simple_tree):
    props = list_properties(simple_tree, level="v")
    assert {"x", "y", "z"}.issubset(props)


def test_bind_and_get_vertex_property(simple_tree):
    g = simple_tree.copy()
    bind_vertex_property(g, "test_val", "double", np.arange(g.num_vertices(), dtype=float))
    assert get_property(g, "test_val", level="v", as_array=True).shape == (17,)


def test_bind_graph_property(simple_tree):
    g = simple_tree.copy()
    bind_graph_property(g, "flag", "bool", True)
    assert bool(get_property(g, "flag", level="g"))


def test_bind_edge_property(graph_with_path_length):
    g = graph_with_path_length.copy()
    data = np.ones(g.num_edges())
    bind_edge_property(g, "weight", "double", data)
    assert get_property(g, "weight", level="e", as_array=True).shape == (16,)


def test_set_property_update_existing(simple_tree):
    g = simple_tree.copy()
    new_r = np.full(g.num_vertices(), 2.0)
    set_property(g, "radius", new_r, level="v")
    assert np.all(get_property(g, "radius", level="v", as_array=True) == 2.0)


def test_set_property_create_new(simple_tree):
    g = simple_tree.copy()
    set_property(g, "custom", np.zeros(g.num_vertices()), level="v", dtype="double", create=True)
    assert "custom" in g.vp


def test_set_property_missing_raises(simple_tree):
    with pytest.raises(KeyError, match="not found"):
        set_property(simple_tree, "nope", [1], level="v")


def test_del_property(simple_tree):
    g = simple_tree.copy()
    bind_vertex_property(g, "temp", "int", np.zeros(g.num_vertices(), dtype=int))
    del_property(g, "temp", level="v")
    assert not g_has_property(g, "temp", "v")


def test_get_property_xyz_components(simple_tree):
    x = get_property(simple_tree, "x", level="v", as_array=True)
    y = get_property(simple_tree, "y", level="v", as_array=True)
    z = get_property(simple_tree, "z", level="v", as_array=True)
    assert x.shape == y.shape == z.shape == (17,)


def test_get_property_search_all_levels(simple_tree):
    val = get_property(simple_tree, "x", as_array=False)
    assert val is not None


def test_get_property_missing_raises(simple_tree):
    with pytest.raises(KeyError):
        get_property(simple_tree, "does_not_exist")


def test_raise_internal_property_missing(simple_tree):
    with pytest.raises(_InternalPropertyMissingError):
        raise_internal_property_missing(simple_tree, "missing_prop", "v")


def test_revert_core_properties_removes_extras(simple_tree):
    g = simple_tree.copy()
    bind_vertex_property(g, "extra", "double", np.zeros(g.num_vertices()))
    bind_graph_property(g, "extra_g", "bool", False)
    revert_core_properties(g)
    assert not g_has_property(g, "extra", "v")
    assert not g_has_property(g, "extra_g", "g")
    assert g_has_property(g, "x", "v")


def test_bind_vector_property_via_set_create(simple_tree):
    g = simple_tree.copy()
    coords = np.stack([g.vp["x"].a, g.vp["y"].a, g.vp["z"].a], axis=1)
    set_property(g, "coords_copy", coords, level="v", dtype="vector<double>", create=True)
    out = get_property(g, "coords_copy", level="v", as_array=True)
    assert out.shape == coords.shape
