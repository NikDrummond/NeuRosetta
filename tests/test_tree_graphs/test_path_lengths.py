"""Tests for tree path-length ops and Forest batching."""

from numpy.testing import assert_allclose

from neurosetta import Forest
from neurosetta.ops.tree_graphs.tree_path_lengths import (
    get_edge_length,
    get_root_distance,
    get_total_cable_length,
)
from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.graph_utils import g_has_property, root_index


def test_get_root_distance_bind_false(test_tree):
    dist = get_root_distance(test_tree, bind=False)
    assert dist is not None
    assert dist.shape == (test_tree.graph.num_vertices(),)
    assert float(dist[root_index(test_tree.graph)]) == 0.0
    assert (dist >= 0).all()


def test_get_root_distance_bind_true(test_tree):
    assert get_root_distance(test_tree, bind=True) is None
    assert g_has_property(test_tree.graph, "Root_distance", "v")
    dist = test_tree.graph.vp["Root_distance"].a
    assert float(dist[root_index(test_tree.graph)]) == 0.0


def test_get_root_distance_creates_missing_edge_lengths(test_tree):
    # _Tree init may already bind Path_length; strip it to test the fallback.
    if "Path_length" in test_tree.graph.ep:
        del test_tree.graph.ep["Path_length"]
    if "Euclidean_length" in test_tree.graph.ep:
        del test_tree.graph.ep["Euclidean_length"]

    assert not g_has_property(test_tree.graph, "Path_length", "e")
    dist = get_root_distance(test_tree, length_type="Path", bind=False)
    assert g_has_property(test_tree.graph, "Path_length", "e")
    assert dist is not None
    assert float(dist[root_index(test_tree.graph)]) == 0.0


def test_get_root_distance_child_equals_parent_plus_edge(test_tree):
    get_edge_length(test_tree, bind=True)
    g = test_tree.graph
    root = root_index(g)
    child = int(g.get_out_neighbors(root)[0])
    edge = g.edge(root, child)
    edge_len = float(g.ep["Path_length"][edge])

    dist = get_root_distance(test_tree, bind=False)
    assert_allclose(dist[child], dist[root] + edge_len)


def test_get_root_distance_euclidean(test_tree):
    dist = get_root_distance(test_tree, length_type="Euclidean", bind=False)
    assert g_has_property(test_tree.graph, "Euclidean_length", "e")
    assert float(dist[root_index(test_tree.graph)]) == 0.0
    assert_allclose(dist, get_root_distance(test_tree, length_type="Path", bind=False))


def test_get_total_cable_length_positive(test_tree):
    assert get_total_cable_length(test_tree) > 0


def test_forest_get_root_distance():
    t1 = make_synthetic_tree(20, tree_id=1, units="nm")
    t2 = make_synthetic_tree(30, tree_id=2, units="nm")
    forest = Forest([t1, t2])

    out = forest.get_root_distance(bind=False, parallel=False)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (t1.count_nodes(),)
    assert out[1].shape == (t2.count_nodes(),)
    assert float(out[0][t1.get_root_index()]) == 0.0
    assert float(out[1][t2.get_root_index()]) == 0.0

    assert forest.get_root_distance(bind=True, parallel=False) is None
    assert "Root_distance" in t1.graph.vp
    assert "Root_distance" in t2.graph.vp


def test_tree_api_get_root_distance():
    tree = make_synthetic_tree(20, tree_id=3, units="nm")
    dist = tree.get_root_distance(bind=False)
    assert dist.shape == (tree.count_nodes(),)
    assert float(dist[tree.get_root_index()]) == 0.0
