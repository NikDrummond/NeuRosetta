"""Strahler order on binary, multifurcating, and reduced trees."""

import numpy as np
from numpy.testing import assert_array_equal
from tree_builders import make_directed_tree

from neurosetta.utils.graph_utils import max_strahler_order, strahler_order
from neurosetta.utils.graph_utils.graph_editing import reduce_graph
from neurosetta.utils.graph_utils.node_types import infer_node_types


def _reducible(g):
    g.vp["radius"] = g.new_vp("double", np.ones(g.num_vertices()))
    g.vp["node_type"] = g.new_vp("int", infer_node_types(g))
    g.gp["ID"] = g.new_gp("long", 7)
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "file_path": ""})
    return g


def test_strahler_catalogue():
    single = make_directed_tree([])
    assert strahler_order(single, bind=False).tolist() == [1]
    assert max_strahler_order(single) == 1

    chain = make_directed_tree([(0, 1), (1, 2)])
    assert strahler_order(chain, bind=False).tolist() == [1, 1, 1]

    balanced = make_directed_tree([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)])
    assert strahler_order(balanced, bind=False).tolist() == [3, 2, 2, 1, 1, 1, 1]

    unbalanced = make_directed_tree([(0, 1), (0, 2), (2, 3), (2, 4)])
    # Child orders at the root are 1 and 2, so the root stays 2.
    assert strahler_order(unbalanced, bind=False).tolist() == [2, 1, 2, 1, 1]

    multi = make_directed_tree([(0, 1), (0, 2), (0, 3)])
    assert strahler_order(multi, bind=False).tolist() == [2, 1, 1, 1]

    shifted = make_directed_tree([(2, 0), (2, 1)], n=3)
    assert strahler_order(shifted, bind=False).tolist() == [1, 1, 2]


def test_transitive_vertices_do_not_change_original_orders():
    base_edges = [(0, 1), (0, 2), (1, 3), (1, 4)]
    base = strahler_order(make_directed_tree(base_edges), bind=False)
    # Vertex 5 is inserted on the edge 0 → 1.
    expanded_edges = [(0, 5), (5, 1), (0, 2), (1, 3), (1, 4)]
    expanded = strahler_order(make_directed_tree(expanded_edges, n=6), bind=False)
    assert expanded[5] == base[1]
    assert_array_equal(expanded[:5], base)


def test_reduction_preserves_strahler_on_surviving_nodes():
    edges = [(0, 4), (4, 1), (1, 2), (1, 3)]
    g = _reducible(make_directed_tree(edges, coords=np.eye(5, 3), path_length=[1, 1, 1, 1]))
    full = strahler_order(g, bind=False)
    reduced = reduce_graph(g)
    reduced_order = strahler_order(reduced, bind=False)
    ids = reduced.vp["ids"].a.astype(int)
    assert_array_equal(reduced_order, full[ids])


def test_strahler_bind_does_not_drop_other_properties():
    g = make_directed_tree([(0, 1), (0, 2)])
    g.vp["keep"] = g.new_vp("int", [3, 4, 5])
    assert strahler_order(g, bind=True) is None
    assert g.vp["strahler_order"].a.tolist() == [2, 1, 1]
    assert g.vp["keep"].a.tolist() == [3, 4, 5]
    assert strahler_order(g, bind=False).tolist() == [2, 1, 1]
