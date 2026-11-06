"""Tests for graph_utils.vertex_inds."""

import pytest
from numpy import unique

from neurosetta.utils.graph_utils.counting import count_branches, count_leaves
from neurosetta.utils.graph_utils.vertex_inds import (
    branch_indices,
    core_indices,
    edge_indices,
    leaf_indices,
    root_index,
    subtree_indices,
)


def test_root_index(simple_tree):
    assert simple_tree.degree_property_map("in").a[root_index(simple_tree)] == 0


def test_leaf_and_branch_indices(simple_tree):
    assert len(leaf_indices(simple_tree)) == count_leaves(simple_tree)
    assert len(branch_indices(simple_tree)) == count_branches(simple_tree)


def test_core_indices_includes_root_by_default(simple_tree):
    core = core_indices(simple_tree, include_root=True)
    assert root_index(simple_tree) in core


def test_core_indices_excludes_root(simple_tree):
    core = core_indices(simple_tree, include_root=False)
    assert root_index(simple_tree) not in core


def test_edge_indices_all_edges(simple_tree):
    edges = edge_indices(simple_tree)
    assert edges.shape == (16, 2)
    for edge in simple_tree.edges():
        index = int(simple_tree.edge_index[edge])
        assert edges[index].tolist() == [int(edge.source()), int(edge.target())]


def test_edge_indices_after_unpurged_deletion(simple_tree):
    """``remove_edge`` leaves index gaps. Plots still need the live edges."""
    removed = next(simple_tree.edges())
    removed_ends = (int(removed.source()), int(removed.target()))
    simple_tree.remove_edge(removed)
    edges = edge_indices(simple_tree)
    assert edges.shape == (15, 2)
    assert removed_ends not in {tuple(row) for row in edges.tolist()}


def test_subtree_indices_from_root(simple_tree):
    root = root_index(simple_tree)
    verts = subtree_indices(simple_tree, root)
    assert len(verts) == 17


def test_subtree_indices_smaller_subtree(simple_tree):
    root = root_index(simple_tree)
    verts = subtree_indices(simple_tree, root + 1)
    assert 0 < len(verts) < 17


def test_edge_indices_invalid_traversal_raises(simple_tree):
    with pytest.raises(ValueError, match="traversal_order"):
        edge_indices(simple_tree, root=0, traversal_order="Invalid")


def test_subtree_indices_leaf_includes_itself(simple_tree):
    leaf = int(leaf_indices(simple_tree)[0])
    assert subtree_indices(simple_tree, leaf).tolist() == [leaf]


def test_subtree_indices_rejects_bad_root(simple_tree):
    with pytest.raises(ValueError):
        subtree_indices(simple_tree, 10_000)
    with pytest.raises(TypeError):
        subtree_indices(simple_tree, 1.5)


def test_subtree_indices_depth_first(simple_tree):
    root = root_index(simple_tree)
    bfs = unique(subtree_indices(simple_tree, root, traversal_order="Breadth"))
    dfs = unique(subtree_indices(simple_tree, root, traversal_order="Depth"))
    assert set(bfs) == set(dfs)
