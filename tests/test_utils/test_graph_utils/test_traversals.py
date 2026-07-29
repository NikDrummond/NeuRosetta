"""Tests for graph_utils.traversals."""

from numpy import unique

from NeuRosetta.utils.graph_utils.traversals import (
    PostOrderVisitor,
    SubtreeMaskVisitor,
    TreeDepthVisitor,
    bf_iterator,
    bfsearch,
    df_iterator,
    dfsearch,
)
from NeuRosetta.utils.graph_utils.vertex_inds import root_index


def test_bf_iterator_returns_edges(simple_tree):
    edges = bf_iterator(simple_tree, root=0, array=True)
    assert edges.shape[1] == 2
    assert len(edges) == 16


def test_df_iterator_returns_edges(simple_tree):
    edges = df_iterator(simple_tree, root=0, array=True)
    assert len(edges) == 16


def test_tree_depth_visitor(graph_with_xyz):
    g = graph_with_xyz.copy()
    root = root_index(g)
    bfsearch(
        g,
        TreeDepthVisitor,
        init_vertex_properties={"depth": "int"},
        root=root,
    )
    assert g.vp["depth"].a[root] == 0
    assert g.vp["depth"].a.max() > 0


def test_subtree_mask_visitor(graph_with_xyz):
    g = graph_with_xyz.copy()
    root = root_index(g)
    bfsearch(
        g,
        SubtreeMaskVisitor,
        init_vertex_properties={"v_subtree_mask": "bool"},
        init_edge_properties={"e_subtree_mask": "bool"},
        root=root + 1,
    )
    assert g.vp["v_subtree_mask"].a.sum() > 0
    assert g.ep["e_subtree_mask"].a.sum() > 0


def test_post_order_visitor(simple_tree):
    vis = dfsearch(simple_tree, PostOrderVisitor, root=root_index(simple_tree))
    assert len(vis.post_order) == 17
    assert len(unique(vis.post_order)) == 17


def test_bfsearch_bind_false_returns_properties(graph_with_xyz):
    g = graph_with_xyz.copy()
    props = bfsearch(
        g,
        TreeDepthVisitor,
        init_vertex_properties={"depth": "int"},
        root=root_index(g),
        bind=False,
    )
    assert "depth" in props
    assert "depth" not in g.vp
