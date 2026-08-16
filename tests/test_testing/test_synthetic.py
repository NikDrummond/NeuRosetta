"""Smoke tests for internal synthetic morphology helpers."""

from neurosetta.ops.tree_graphs import count_nodes, count_roots, get_total_cable_length
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


def test_make_synthetic_tree():
    tree = make_synthetic_tree(n=25, seed=0)
    assert tree.metadata["synthetic"] is True
    assert count_nodes(tree) == 25
    assert count_roots(tree) == 1
    assert get_total_cable_length(tree) > 0


def test_make_synthetic_forest():
    forest = make_synthetic_forest(n_trees=2, n=10, seed=1)
    assert len(forest) == 2
    assert forest.ids() == [1, 2]
    assert all(t.metadata["synthetic"] for t in forest)
