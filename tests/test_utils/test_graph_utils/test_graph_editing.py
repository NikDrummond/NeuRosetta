"""Tests for graph_utils.graph_editing."""

from NeuRosetta.utils.graph_utils.counting import count_vertices
from NeuRosetta.utils.graph_utils.graph_editing import reduce_graph, reroot_graph
from NeuRosetta.utils.graph_utils.vertex_inds import root_index


def test_reduce_graph_fewer_vertices(reduced_graph_fixture):
    g = reduced_graph_fixture.copy()
    n_before = count_vertices(g)
    g_red = reduce_graph(g)
    assert count_vertices(g_red) < n_before
    assert g_red.gp["metadata"]["isReduced"] is True


def test_reduce_graph_preserves_id(reduced_graph_fixture):
    g = reduced_graph_fixture.copy()
    g_red = reduce_graph(g)
    assert int(g_red.gp["ID"]) == 42


def test_reroot_graph_same_vertex_count(reduced_graph_fixture):
    g = reduced_graph_fixture.copy()
    new_root = root_index(g) + 1
    g_new = reroot_graph(g, new_root)
    assert count_vertices(g_new) == count_vertices(g)


def test_reroot_graph_root_has_zero_in_degree(reduced_graph_fixture):
    g = reduced_graph_fixture.copy()
    new_root = root_index(g) + 1
    g_new = reroot_graph(g, new_root)
    in_deg = g_new.degree_property_map("in").a
    assert (in_deg == 0).sum() == 1
