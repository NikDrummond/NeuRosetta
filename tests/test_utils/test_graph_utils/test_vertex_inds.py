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


def test_subtree_indices_depth_first(simple_tree):
    root = root_index(simple_tree)
    bfs = unique(subtree_indices(simple_tree, root, traversal_order="Breadth"))
    dfs = unique(subtree_indices(simple_tree, root, traversal_order="Depth"))
    assert set(bfs) == set(dfs)
