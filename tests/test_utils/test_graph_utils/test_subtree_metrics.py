"""Subtree counts, cable, depth, and preserved score / asymmetry definitions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from tree_builders import make_directed_tree

from neurosetta.utils.graph_utils import (
    extract_subgraph,
    max_subgraph_ind,
    partition_asymmetry,
    subgraph_branch_count,
    subgraph_branch_depth,
    subgraph_depth,
    subgraph_edge_count,
    subgraph_height,
    subgraph_leaf_count,
    subgraph_mean_leaf_distance,
    subgraph_path_length,
    subgraph_score,
    subgraph_vertex_count,
)
from neurosetta.utils.graph_utils.subgraphs import subgraph_score as score_from_facade
from neurosetta.utils.graph_utils.vertex_inds import leaf_indices

# 0 —1→ 1 (leaf)
#  └2→ 2 —1→ 3 (leaf)
#       └3→ 4 (leaf)
CABLE_EDGES = [(0, 1), (0, 2), (2, 3), (2, 4)]
CABLE_LENGTHS = [1.0, 2.0, 1.0, 3.0]


def _cable():
    return make_directed_tree(CABLE_EDGES, path_length=CABLE_LENGTHS)


def test_existing_subgraph_imports():
    assert score_from_facade is subgraph_score
    assert callable(extract_subgraph)
    assert callable(max_subgraph_ind)


def test_single_vertex_counts_and_depth():
    g = make_directed_tree([])
    assert subgraph_leaf_count(g, bind=False).tolist() == [1]
    assert subgraph_edge_count(g, bind=False).tolist() == [0]
    assert subgraph_vertex_count(g, bind=False).tolist() == [1]
    assert subgraph_branch_count(g, bind=False).tolist() == [0]
    assert subgraph_depth(g, bind=False).tolist() == [0]
    assert subgraph_branch_depth(g, bind=False).tolist() == [0]
    g.ep["Path_length"] = g.new_ep("double")
    assert subgraph_path_length(g, bind=False).tolist() == [0.0]
    assert subgraph_height(g, bind=False).tolist() == [0.0]
    assert subgraph_mean_leaf_distance(g, bind=False).tolist() == [0.0]


def test_chain_metrics():
    g = make_directed_tree([(0, 1), (1, 2)], path_length=[1.0, 1.0])
    assert subgraph_vertex_count(g, bind=False).tolist() == [3, 2, 1]
    assert subgraph_edge_count(g, bind=False).tolist() == [2, 1, 0]
    assert subgraph_leaf_count(g, bind=False).tolist() == [1, 1, 1]
    assert subgraph_branch_count(g, bind=False).tolist() == [0, 0, 0]
    assert subgraph_depth(g, bind=False).tolist() == [2, 1, 0]
    assert subgraph_path_length(g, bind=False).tolist() == [2.0, 1.0, 0.0]
    assert subgraph_height(g, bind=False).tolist() == [2.0, 1.0, 0.0]
    assert subgraph_mean_leaf_distance(g, bind=False).tolist() == [2.0, 1.0, 0.0]


def test_cable_tree_metrics_and_identity():
    g = _cable()
    leaves = subgraph_leaf_count(g, bind=False)
    edges = subgraph_edge_count(g, bind=False)
    verts = subgraph_vertex_count(g, bind=False)
    branches = subgraph_branch_count(g, bind=False)
    cable = subgraph_path_length(g, bind=False)
    assert leaves.tolist() == [3, 1, 2, 1, 1]
    assert edges.tolist() == [4, 0, 2, 0, 0]
    assert verts.tolist() == [5, 1, 3, 1, 1]
    assert branches.tolist() == [2, 0, 1, 0, 0]
    assert_allclose(cable, [7, 0, 4, 0, 0])
    assert np.array_equal(verts, edges + 1)
    assert_allclose(cable[0], np.sum(CABLE_LENGTHS))
    leaf = int(leaf_indices(g)[0])
    assert verts[leaf] == 1
    assert edges[leaf] == 0
    assert leaves[leaf] == 1
    assert cable[leaf] == 0
    assert subgraph_depth(g, bind=False).tolist() == [2, 0, 1, 0, 0]
    assert subgraph_branch_depth(g, bind=False).tolist() == [1, 0, 0, 0, 0]
    assert_allclose(subgraph_height(g, bind=False), [5, 0, 3, 0, 0])
    assert_allclose(subgraph_mean_leaf_distance(g, bind=False), [3, 0, 2, 0, 0])


def test_branch_count_includes_root_when_it_branches():
    g = make_directed_tree([(0, 1), (0, 2)])
    assert subgraph_branch_count(g, bind=False).tolist() == [1, 0, 0]


def test_bind_names_and_no_mutation_when_false():
    g = _cable()
    before = set(g.vp.keys())
    subgraph_leaf_count(g, bind=False)
    assert set(g.vp.keys()) == before
    assert subgraph_leaf_count(g, bind=True) is None
    assert "subtree_leaf_count" in g.vp
    assert "subgraph_score" not in g.vp


def test_euclidean_property_name_does_not_overwrite_path_cable():
    g = _cable()
    g.ep["Euclidean_length"] = g.new_ep("double", [10.0, 10.0, 10.0, 10.0])
    subgraph_path_length(g, bind=True)
    subgraph_path_length(g, bind=True, length_property="Euclidean_length")
    assert_allclose(g.vp["subtree_path_length"].a, [7, 0, 4, 0, 0])
    assert_allclose(g.vp["subtree_path_length_euclidean"].a, [40, 0, 20, 0, 0])


def test_missing_path_length_is_not_replaced():
    g = make_directed_tree([(0, 1)])
    with pytest.raises(ValueError, match="Path_length"):
        subgraph_path_length(g, bind=False)


def test_nonzero_root_index():
    g = make_directed_tree([(2, 0), (2, 1)], n=3, path_length=[1.0, 4.0])
    assert subgraph_vertex_count(g, bind=False).tolist() == [1, 1, 3]
    assert subgraph_leaf_count(g, bind=False).tolist() == [1, 1, 2]
    assert_allclose(subgraph_path_length(g, bind=False), [0, 0, 5])


def test_subgraph_score_matches_definition_and_zero_cable():
    g = _cable()
    score = subgraph_score(g, bind=False)
    # Branches are 0 and 2. total cable 7, total leaves 3.
    # s(0) = 1 - 7/7 + 3/3 = 1
    # s(2) = 1 - 4/7 + 2/3
    assert score[1] == 0 and score[3] == 0 and score[4] == 0
    assert_allclose(score[0], 1.0)
    assert_allclose(score[2], 1.0 - 4.0 / 7.0 + 2.0 / 3.0)
    assert max_subgraph_ind(g) == int(np.argmax(score))

    flat = make_directed_tree([(0, 1), (0, 2)], path_length=[0.0, 0.0])
    flat_score = subgraph_score(flat, bind=False)
    assert_allclose(flat_score, [1.0, 0.0, 0.0])


def test_partition_asymmetry_regression():
    g = _cable()
    unweighted = partition_asymmetry(g, mode="unweighted", bind=False)
    weighted = partition_asymmetry(g, mode="weighted", bind=False)
    van_pelt = partition_asymmetry(g, mode="van_pelt", bind=False)
    cable = partition_asymmetry(g, mode="cable", bind=False)
    # Node 2: leaves 1, 1. Node 0: leaves 1 and 2.
    assert_allclose(unweighted, [0.5, 0, 0, 0, 0])
    assert_allclose(weighted, [0.5, 0, 0, 0, 0])
    assert_allclose(van_pelt, [1.0, 0, 0, 0, 0])
    # Daughter cables at 0: 1 and 6. At 2: 1 and 3.
    assert_allclose(cable[0], 5.0 / 7.0)
    assert_allclose(cable[2], 0.5)
    assert cable[1] == 0

    hidden = partition_asymmetry(g, mode="unweighted", bind=False, include_root=False)
    assert hidden[0] == 0
    assert partition_asymmetry(g, mode="cable", bind=True) is None
    assert "partition_asymmetry_cable" in g.vp


def test_multifurcation_asymmetry_averages_pairs():
    # Root with three children whose leaf counts are 1, 1, and 2.
    edges = [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)]
    g = make_directed_tree(edges, path_length=[1, 1, 1, 1, 1])
    pa = partition_asymmetry(g, mode="unweighted", bind=False)
    # Pairs (1,1)=0, (1,2)=0.5, (1,2)=0.5. Mean 1/3.
    assert_allclose(pa[0], 1.0 / 3.0)


def test_invalid_trees_rejected():
    empty = make_directed_tree([], n=0)
    with pytest.raises(ValueError, match="empty"):
        subgraph_leaf_count(empty, bind=False)
    two = make_directed_tree([(0, 1), (2, 3)], n=4)
    with pytest.raises(ValueError, match="one root"):
        subgraph_depth(two, bind=False)
    undirected = make_directed_tree([(0, 1)])
    undirected.set_directed(False)
    with pytest.raises(ValueError, match="directed"):
        subgraph_vertex_count(undirected, bind=False)


def test_active_filter_rejected(simple_tree):
    mask = simple_tree.new_vp("bool", val=True)
    simple_tree.set_vertex_filter(mask)
    with pytest.raises(ValueError, match="filter"):
        subgraph_leaf_count(simple_tree, bind=False)
    simple_tree.clear_filters()
