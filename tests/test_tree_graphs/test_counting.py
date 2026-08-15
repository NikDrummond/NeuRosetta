# tests/test_tree_graphs/test_counting.py
from neurosetta.ops.tree_graphs.tree_counting import (
    count_branches,
    count_edges,
    count_leaves,
    count_nodes,
    count_roots,
)


def test_count_nodes(test_tree):
    assert count_nodes(test_tree) == 17


def test_count_edges(test_tree):
    assert count_edges(test_tree) == 16


def test_count_roots(test_tree):
    assert count_roots(test_tree) == 1


def test_count_leaves(test_tree):
    assert count_leaves(test_tree) == 6


def test_count_branches(test_tree):
    assert count_branches(test_tree) == 4
