"""Tests for graph_utils.counting."""

from neurosetta.utils.graph_utils.counting import (
    count_branches,
    count_edges,
    count_leaves,
    count_roots,
    count_transitive_vertices,
    count_vertices,
)


def test_count_vertices(simple_tree):
    assert count_vertices(simple_tree) == 17


def test_count_edges(simple_tree):
    assert count_edges(simple_tree) == 16


def test_count_roots(simple_tree):
    assert count_roots(simple_tree) == 1


def test_count_leaves(simple_tree):
    assert count_leaves(simple_tree) == 6


def test_count_branches(simple_tree):
    assert count_branches(simple_tree) == 4


def test_count_transitive_vertices(simple_tree):
    assert count_transitive_vertices(simple_tree) == 7


def test_count_roots_zero_one_and_many():
    from graph_tool.all import Graph

    empty = Graph(directed=True)
    assert count_roots(empty) == 0

    one = Graph(directed=True)
    one.add_vertex(1)
    assert count_roots(one) == 1

    many = Graph(directed=True)
    many.add_vertex(4)
    many.add_edge_list([(0, 1), (2, 3)])
    assert count_roots(many) == 2
