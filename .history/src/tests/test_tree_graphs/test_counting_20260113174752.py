# tests/test_tree_graphs/test_counting.py
import pytest
from NeuRosetta.tree_graphs.counting import (
    count_vertices, count_edges, count_roots, count_leaves, count_branches
)

def test_count_vertices(test_tree):
    assert count_vertices(test_tree) == 17

def test_count_edges(test_tree):
    assert count_edges(test_tree) == 16

def test_count_roots(test_tree):
    assert count_roots(test_tree) == 1

def test_count_leaves(test_tree):
    assert count_leaves(test_tree) == 6

def test_count_branches(test_tree):
    assert count_branches(test_tree) == 11
