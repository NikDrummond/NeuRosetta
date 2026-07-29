"""Shared fixtures for utils unit tests."""

import pytest
from numpy import array, ones_like, sqrt


@pytest.fixture
def graph_with_xyz(simple_tree):
    """Graph with separate x/y/z vertex properties derived from coordinates."""
    g = simple_tree
    coords = g.vp["coordinates"].get_2d_array().T
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    return g


@pytest.fixture
def graph_with_path_length(graph_with_xyz):
    """Graph with Euclidean Path_length on each edge."""
    from NeuRosetta.utils.graph_utils.coordinates import edge_coordinates

    g = graph_with_xyz
    p1, p2 = edge_coordinates(g)
    lengths = sqrt(((p2 - p1) ** 2).sum(axis=1))
    g.ep["Path_length"] = g.new_ep("double", lengths)
    return g


@pytest.fixture
def reduced_graph_fixture(graph_with_path_length):
    """Graph with metadata/ID for reduce_graph and reroot_graph."""
    g = graph_with_path_length
    g.gp["ID"] = g.new_gp("long", 42)
    g.gp["metadata"] = g.new_gp("object", {"file_path": "/tmp/test.swc", "isReduced": False})
    return g
