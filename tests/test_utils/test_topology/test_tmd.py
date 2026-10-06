"""Tests for DFS TMD on original graph indices."""

from __future__ import annotations

import numpy as np
import pytest
from graph_tool.all import Graph
from helpers import make_random_rooted_tree, reference_tmd_pairs

from neurosetta.utils.graph_utils import root_index
from neurosetta.utils.topology import tmd
from neurosetta.utils.topology.visitors import TMDVisitor


def _pair_set(result: dict) -> set[tuple[int, int]]:
    return set(zip(result["birth_ind"].tolist(), result["death_ind"].tolist(), strict=True))


def test_single_chain(chain_graph):
    result = tmd(chain_graph, "f")
    assert _pair_set(result) == {(0, 3)}
    assert result["birth_ind"].tolist() == [0]
    assert result["death_ind"].tolist() == [3]
    np.testing.assert_allclose(result["birth"], [0.0])
    np.testing.assert_allclose(result["death"], [3.0])
    np.testing.assert_allclose(result["survival_len"], [3.0])


def test_binary_tree(binary_graph):
    result = tmd(binary_graph, "f")
    # Expected: (1, 4) loser at branch 1; (0, 2) loser at root; (0, 3) final
    assert _pair_set(result) == {(1, 4), (0, 2), (0, 3)}
    assert 3 in result["death_ind"]  # final survivor leaf
    # root pairs with final survivor
    root_pairs = [
        (b, d) for b, d in zip(result["birth_ind"], result["death_ind"], strict=True) if b == 0
    ]
    assert (0, 3) in root_pairs


def test_multifurcation(multifurcation_graph):
    result = tmd(multifurcation_graph, "f")
    # leaves 1,2,3 values 2,5,3 → survivor 2; losers 1 and 3 at root; final (0,2)
    assert _pair_set(result) == {(0, 1), (0, 3), (0, 2)}


def test_asymmetric_tree(asymmetric_graph):
    result = tmd(asymmetric_graph, "f")
    ref = reference_tmd_pairs(asymmetric_graph, asymmetric_graph.vp["f"].a, root=0)
    assert _pair_set(result) == set(ref)


def test_root_with_several_children(multifurcation_graph):
    result = tmd(multifurcation_graph, "f")
    assert all(b == 0 for b in result["birth_ind"])
    assert len(result["birth_ind"]) == 3  # 2 losers + final


def test_single_vertex_tree():
    g = Graph(directed=True)
    g.add_vertex()
    g.vp["f"] = g.new_vp("double", [1.5])
    result = tmd(g, "f")
    assert _pair_set(result) == {(0, 0)}
    np.testing.assert_allclose(result["survival_len"], [0.0])


def test_degree1_continuation_no_extra_events(chain_graph):
    result = tmd(chain_graph, "f")
    # only root-final pair; vertices 1,2 never appear as birth_ind
    assert set(result["birth_ind"].tolist()) == {0}


def test_original_vertex_indices_preserved(binary_graph):
    result = tmd(binary_graph, "f")
    n = binary_graph.num_vertices()
    assert result["birth_ind"].max() < n
    assert result["death_ind"].max() < n
    # known leaf indices
    assert set(result["death_ind"].tolist()).issubset({2, 3, 4})


def test_final_survivor_pairs_with_root(binary_graph):
    result = tmd(binary_graph, "f")
    # max leaf value is 5 at vertex 3
    assert (0, 3) in _pair_set(result)


def test_bind_stores_graph_property(chain_graph):
    result = tmd(chain_graph, "f", bind=True)
    assert "TMD" in chain_graph.gp
    assert chain_graph.gp["TMD"]["death_ind"].tolist() == result["death_ind"].tolist()


def test_func_property_map(chain_graph):
    result = tmd(chain_graph, chain_graph.vp["f"])
    assert _pair_set(result) == {(0, 3)}


def test_bad_func_type(chain_graph):
    with pytest.raises(TypeError):
        tmd(chain_graph, 123)  # type: ignore[arg-type]


def test_matches_reference_on_random_trees():
    for seed in range(20):
        g, values = make_random_rooted_tree(n=40, seed=seed)
        root = root_index(g)
        result = tmd(g, "f")
        ref = set(reference_tmd_pairs(g, values, root=root))
        assert _pair_set(result) == ref


def test_no_reduction_dependency(asymmetric_graph):
    """TMD runs on the original graph and retains original leaf indices."""
    full = tmd(asymmetric_graph, "f")
    assert isinstance(TMDVisitor, type)
    leaves_full = {int(v) for v in asymmetric_graph.vertices() if v.out_degree() == 0}
    assert set(full["death_ind"].tolist()).issubset(leaves_full)
    # Continuation vertex 1 is never a birth index (only branch 3 and root 0).
    assert 1 not in set(full["birth_ind"].tolist())
