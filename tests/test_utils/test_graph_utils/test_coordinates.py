"""Tests for graph_utils.coordinates."""

import pytest
from numpy import allclose

from NeuRosetta.utils.graph_utils.coordinates import (
    edge_coordinates,
    edge_coordinates_subtree,
    vertex_coordinates,
    vertex_coordinates_subtree,
)
from NeuRosetta.utils.graph_utils.vertex_inds import root_index


def test_vertex_coordinates_shape(graph_with_xyz):
    coords = vertex_coordinates(graph_with_xyz)
    assert coords.shape == (17, 3)


def test_vertex_coordinates_soa(graph_with_xyz):
    coords = vertex_coordinates(graph_with_xyz, SoA=True)
    assert coords.shape == (3, 17)


def test_vertex_coordinates_subset(graph_with_xyz):
    subset = [0, 1, 2]
    coords = vertex_coordinates(graph_with_xyz, subset=subset)
    assert coords.shape == (3, 3)


def test_vertex_coordinates_subtree(graph_with_xyz):
    root = root_index(graph_with_xyz)
    coords = vertex_coordinates_subtree(graph_with_xyz, root)
    assert coords.shape[0] == 17


def test_edge_coordinates(graph_with_xyz):
    p1, p2 = edge_coordinates(graph_with_xyz)
    assert p1.shape == p2.shape == (16, 3)
    assert not allclose(p1, p2)


def test_edge_coordinates_subtree(graph_with_xyz):
    root = root_index(graph_with_xyz)
    p1, p2 = edge_coordinates_subtree(graph_with_xyz, root)
    assert p1.shape[0] == 16


def test_vertex_coordinates_missing_property_raises(simple_tree):
    with pytest.raises(Exception, match="Internal Property Missing"):
        vertex_coordinates(simple_tree)
