"""Tests for graph_utils.paths."""

import pytest
from numpy.testing import assert_allclose

from neurosetta.utils.graph_utils.gt_properties import _InternalPropertyMissingError
from neurosetta.utils.graph_utils.paths import root_distance
from neurosetta.utils.graph_utils.vertex_inds import root_index


def test_root_distance_root_is_zero(graph_with_path_length):
    dist = root_distance(graph_with_path_length)
    assert float(dist.a[root_index(graph_with_path_length)]) == 0.0


def test_root_distance_nonnegative(graph_with_path_length):
    dist = root_distance(graph_with_path_length).a
    assert dist.shape == (graph_with_path_length.num_vertices(),)
    assert (dist >= 0).all()


def test_root_distance_matches_edge_to_child(graph_with_path_length):
    g = graph_with_path_length
    root = root_index(g)
    # First outgoing edge from root: parent → child
    child = int(g.get_out_neighbors(root)[0])
    edge = g.edge(root, child)
    expected = float(g.ep["Path_length"][edge])
    dist = root_distance(g).a
    assert_allclose(dist[child], expected)


def test_root_distance_missing_property_raises(simple_tree):
    with pytest.raises(_InternalPropertyMissingError):
        root_distance(simple_tree, ep="Path_length")
