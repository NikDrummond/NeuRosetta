# tests/test_tree_graphs/test_counting.py
from NeuRosetta.ops.tree_graphs.counting import (
    count_tree_branches,
    count_tree_edges,
    count_tree_leaves,
    count_tree_nodes,
    count_tree_roots,
)


def test_count_nodes(test_tree):
    assert count_tree_nodes(test_tree) == 17


def test_count_edges(test_tree):
    assert count_tree_edges(test_tree) == 16


def test_count_roots(test_tree):
    assert count_tree_roots(test_tree) == 1


def test_count_leaves(test_tree):
    assert count_tree_leaves(test_tree) == 6


def test_count_branches(test_tree):
    assert count_tree_branches(test_tree) == 4
