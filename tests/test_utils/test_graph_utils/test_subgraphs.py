"""Tests for graph_utils.subgraphs."""

import numpy as np

from neurosetta.utils.graph_utils.subgraphs import (
    extract_subgraph,
    max_subgraph_ind,
    partition_asymmetry,
    subgraph_score,
)
from neurosetta.utils.graph_utils.traversals import SubtreeMaskVisitor, bfsearch
from neurosetta.utils.graph_utils.vertex_inds import root_index


def test_subgraph_score_bind(graph_with_path_length):
    g = graph_with_path_length.copy()
    subgraph_score(g, bind=True)
    assert "subgraph_score" in g.vp
    assert g.vp["subgraph_score"].a.sum() >= 0


def test_subgraph_score_returns_array(graph_with_path_length):
    g = graph_with_path_length.copy()
    scores = subgraph_score(g, bind=False)
    assert scores.shape == (17,)
    assert np.isfinite(scores).all()


def test_max_subgraph_ind(graph_with_path_length):
    g = graph_with_path_length.copy()
    idx = max_subgraph_ind(g)
    assert 0 <= idx < g.num_vertices()


def test_partition_asymmetry_unweighted(graph_with_path_length):
    g = graph_with_path_length.copy()
    pa = partition_asymmetry(g, weighted=False, bind=False)
    assert pa.shape == (17,)
    assert pa[g.get_out_degrees(g.get_vertices()) >= 2].sum() >= 0


def test_partition_asymmetry_weighted(graph_with_path_length):
    g = graph_with_path_length.copy()
    pa = partition_asymmetry(g, weighted=True, bind=False)
    assert pa.shape == (17,)


def test_extract_subgraph(graph_with_path_length):
    g = graph_with_path_length.copy()
    root = root_index(g)
    bfsearch(
        g,
        SubtreeMaskVisitor,
        init_vertex_properties={"v_subtree_mask": "bool"},
        init_edge_properties={"e_subtree_mask": "bool"},
        root=root + 1,
    )
    n_before = g.num_vertices()
    extract_subgraph(g, revert_properties=False)
    assert g.num_vertices() < n_before
