"""Tests for synapse transfer through tree reduction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from graph_tool.all import Graph

from neurosetta.api import Tree
from neurosetta.utils.graph_utils import ReductionMap, reduce_graph

# Unit-unset warnings are covered in test_synapse_units.py.
pytestmark = pytest.mark.filterwarnings("ignore:.*units.*:UserWarning")


def _chain_tree() -> Tree:
    """A --e0(len=2)-- B --e1(len=3)-- C --e2(len=5)-- D  (B,C transitive)."""
    g = Graph(directed=True)
    coords = np.array(
        [
            [0.0, 0.0, 0.0],  # A
            [2.0, 0.0, 0.0],  # B
            [5.0, 0.0, 0.0],  # C
            [10.0, 0.0, 0.0],  # D
        ]
    )
    for _ in range(4):
        g.add_vertex()
    g.add_edge_list([(0, 1), (1, 2), (2, 3)])
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    g.vp["radius"] = g.new_vp("double", np.ones(4))
    g.vp["node_type"] = g.new_vp("int", [-1, 0, 0, 6])
    g.ep["Path_length"] = g.new_ep("double", [2.0, 3.0, 5.0])
    g.gp["ID"] = g.new_gp("long", 7)
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "Flag": False})
    return Tree.from_graph(g)


def test_reduce_graph_return_mapping():
    tree = _chain_tree()
    g_red, rmap = reduce_graph(tree.graph, return_mapping=True)
    assert isinstance(rmap, ReductionMap)
    assert rmap.n_reduced_edges == 1
    assert rmap.section_edge_indices[0] == (0, 1, 2)
    assert rmap.section_edge_lengths[0] == pytest.approx((2.0, 3.0, 5.0))
    assert g_red.num_edges() == 1
    assert float(g_red.ep["Path_length"].a[0]) == pytest.approx(10.0)


def test_synapse_survives_reduction_cable_positions():
    tree = _chain_tree()
    # Midpoints of e0, e1, e2 + endpoint on e2 target.
    syn = pd.DataFrame(
        {
            "synapse_id": [1, 2, 3, 4],
            "type": ["pre", "post", "pre", "post"],
            "x": [1.0, 3.5, 7.5, 10.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "partner_id": [10, 20, 10, 30],
        }
    )
    tree.set_synapses(syn)
    tree.map_synapses()
    n_before = len(tree.synapses)
    before_ids = tree.synapses.to_dataframe(copy=False)["synapse_id"].to_numpy()
    before_raw = tree.synapses.coordinates.copy()
    before_mapped = tree.synapses.mapped_coordinates.copy()
    before_dist = tree.synapses.to_dataframe(copy=False)["distance_to_tree"].to_numpy()
    before_path = tree.get_synapse_path_distance().copy()

    tree.get_reduced_tree(inplace=True)

    assert len(tree.synapses) == n_before
    df = tree.synapses.to_dataframe(copy=False)
    np.testing.assert_array_equal(df["synapse_id"], before_ids)
    np.testing.assert_allclose(tree.synapses.coordinates, before_raw)
    np.testing.assert_allclose(tree.synapses.mapped_coordinates, before_mapped)
    np.testing.assert_allclose(df["distance_to_tree"], before_dist)
    assert tree.synapses.mapping_valid
    assert np.all(df["edge_index"] == 0)

    # Section distances: mid e0 → 1; mid e1 → 2+1.5=3.5; mid e2 → 2+3+2.5=7.5; end → 10
    expected_along = np.array([1.0, 3.5, 7.5, 10.0])
    np.testing.assert_allclose(df["distance_along_edge"], expected_along, atol=1e-6)
    np.testing.assert_allclose(df["edge_fraction"], expected_along / 10.0, atol=1e-6)

    after_path = tree.get_synapse_path_distance()
    np.testing.assert_allclose(before_path, after_path, atol=1e-6)


def test_unmapped_synapses_preserved_on_reduce():
    tree = _chain_tree()
    tree.set_synapses(
        pd.DataFrame(
            {
                "synapse_id": [1],
                "type": ["pre"],
                "x": [1.0],
                "y": [0.0],
                "z": [0.0],
                "partner_id": [9],
            }
        )
    )
    tree.get_reduced_tree(inplace=True)
    assert tree.synapses is not None
    assert len(tree.synapses) == 1
    assert not tree.synapses.is_mapped


def test_reduced_path_distance_when_edge_order_is_permuted():
    """Cable to a synapse survives reduction on a hashed multi-edge graph.

    ``Graph(..., hashed=True)`` does not iterate edges in edge-index order.
    Synapse ``edge_index`` is an edge index. Indexing ``get_edges()`` with
    that value used to move root cable distance after reduction.
    """
    g = Graph(directed=True)
    # Three sections so the reduced graph has three edges.
    # 0-1-2 leaf, 0-3 leaf, 0-4-5 leaf. Lengths are not uniform.
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 3.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 5.0],
        ]
    )
    g.add_vertex(6)
    g.add_edge_list([(0, 1), (1, 2), (0, 3), (0, 4), (4, 5)])
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    g.vp["radius"] = g.new_vp("double", np.ones(6))
    g.vp["node_type"] = g.new_vp("int", [-1, 0, 6, 6, 0, 6])
    g.gp["ID"] = g.new_gp("long", 3)
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "Flag": False})
    tree = Tree.from_graph(g)
    tree.set_units("micron")
    # Off the first segment, so the stored fraction is not an endpoint.
    tree.set_synapses(
        pd.DataFrame(
            {
                "synapse_id": [1, 2],
                "type": ["pre", "post"],
                "x": [1.2, 0.0],
                "y": [1.0, 2.0],
                "z": [0.0, 0.0],
                "partner_id": [8, 9],
            }
        )
    )
    tree.map_synapses(max_distance=None)
    before = tree.get_synapse_path_distance().copy()
    # (1.2, 1, 0) projects onto (1,0,0)->(1,3,0) at cable 1 from that node.
    # That node is already 1 from the root, so the synapse is at cable 2.
    # (0, 2, 0) is the end of the length-2 edge out of the root.
    np.testing.assert_allclose(before, [2.0, 2.0], atol=1e-9)

    tree.get_reduced_tree(inplace=True)
    assert tree.count_edges() == 3
    after = tree.get_synapse_path_distance()
    np.testing.assert_allclose(after, before, atol=1e-9)

    # Each reduced edge's stored cable matches the property read by descriptor,
    # and edge_indices row i is that same edge.
    reduced_edges = tree.get_edge_indices()
    stored = np.asarray(tree.graph.ep["Path_length"].a, dtype=float)
    for edge in tree.graph.edges():
        index = int(tree.graph.edge_index[edge])
        assert int(reduced_edges[index, 0]) == int(edge.source())
        assert int(reduced_edges[index, 1]) == int(edge.target())
        assert stored[index] == pytest.approx(float(tree.graph.ep["Path_length"][edge]))


def test_branched_reduction_separate_sections():
    """Root with two leaves: two reduced edges; synapses stay on their section."""
    g = Graph(directed=True)
    # 0 root -- 1 transitive -- 2 leaf
    #          \- 3 leaf
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ]
    )
    for _ in range(4):
        g.add_vertex()
    g.add_edge_list([(0, 1), (1, 2), (0, 3)])
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    g.vp["radius"] = g.new_vp("double", np.ones(4))
    g.vp["node_type"] = g.new_vp("int", [-1, 0, 6, 6])
    g.ep["Path_length"] = g.new_ep("double", [1.0, 1.0, 2.0])
    g.gp["ID"] = g.new_gp("long", 8)
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "Flag": False})
    tree = Tree.from_graph(g)

    # On first section midpoint and on direct leaf edge midpoint.
    tree.set_synapses(
        pd.DataFrame(
            {
                "synapse_id": [1, 2],
                "type": ["pre", "post"],
                "x": [0.5, 0.0],
                "y": [0.0, 1.0],
                "z": [0.0, 0.0],
                "partner_id": [1, 2],
            }
        )
    )
    tree.map_synapses()
    edges_before = tree.synapses.to_dataframe(copy=False)["edge_index"].to_numpy()
    assert edges_before[0] != edges_before[1]
    tree.get_reduced_tree(inplace=True)
    edges_after = tree.synapses.to_dataframe(copy=False)["edge_index"].to_numpy()
    assert edges_after[0] != edges_after[1]
    assert tree.count_edges() == 2
