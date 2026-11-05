"""Tree and Forest bindings for path queries."""

import numpy as np
import pytest
from graph_tool.all import Graph
from numpy.testing import assert_allclose, assert_array_equal

import neurosetta as nr
from neurosetta.api import Forest, Tree

_EDGES = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (4, 6), (4, 7)]
_PATH = [1.0, 2.0, 1.0, 1.0, 5.0, 6.0, 7.0]
_EUCL = [10.0, 3.0, 10.0, 4.0, 8.0, 7.0, 9.0]
_COORDS = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (2.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 2.0, 0.0),
    (1.0, 2.0, 0.0),
    (1.0, -1.0, 0.0),
]


def _neuron(tree_id=1):
    g = Graph(directed=True)
    g.add_vertex(len(_COORDS))
    g.add_edge_list(np.asarray(_EDGES, dtype=int))
    xyz = np.asarray(_COORDS, dtype=float)
    g.vp["x"] = g.new_vp("double", xyz[:, 0])
    g.vp["y"] = g.new_vp("double", xyz[:, 1])
    g.vp["z"] = g.new_vp("double", xyz[:, 2])
    g.ep["Path_length"] = g.new_ep("double", _PATH)
    g.ep["Euclidean_length"] = g.new_ep("double", _EUCL)
    return Tree(ID=tree_id, metadata={"isReduced": False}, graph=g)


def test_tree_paths_match_the_reference_tree():
    tree = _neuron()
    assert_array_equal(nr.get_path(tree, 0, 6), [0, 1, 4, 6])
    assert_array_equal(tree.get_path(0, 6), [0, 1, 4, 6])
    assert tree.get_path(6, 0) is None
    assert_array_equal(tree.get_path(6, 3, directed=False), [6, 4, 1, 3])
    assert_array_equal(tree.get_path_edges(3, 4, directed=False), [[3, 1], [1, 4]])
    assert tree.path_exists(1, 7) is True
    assert tree.path_exists(7, 1) is False
    assert tree.get_path_length(0, 6) == 8.0
    assert tree.get_path_edge_count(0, 6) == 3
    assert_allclose(tree.get_path_cumulative_length(0, 6), [0.0, 1.0, 2.0, 8.0])
    assert tree.get_parent_index(0) is None
    assert tree.get_parent_index(7) == 4
    assert tree.is_ancestor(4, 6) is True
    assert tree.is_ancestor(6, 6) is True
    assert tree.get_lowest_common_ancestor(6, 7) == 4
    assert tree.get_lowest_common_ancestor(3, 6, 7) == 1
    assert tree.get_lowest_common_ancestor([3, 4, 6, 7]) == 1
    assert tree.get_lowest_common_ancestor(3, 5, 7) == 0
    assert tree.get_lowest_common_ancestor([6, 7, 4]) == 4
    assert tree.get_lowest_common_ancestor([6]) == 6
    assert tree.get_lowest_common_ancestor([6, 6, 6]) == 6
    assert tree.get_lowest_common_ancestor([0, 6, 7]) == 0
    with pytest.raises(ValueError):
        tree.get_lowest_common_ancestor([])
    assert_allclose(tree.get_path_tortuosity(0, 3), 1.0)
    assert_allclose(tree.get_path_euclidean_distance(0, 3), 2.0)
    bound = tree.get_source_distances(1, bind=True)
    assert bound is None
    assert "path_distance_from_1_auto_directed" in tree.graph.vp
    assert "Root_distance" not in tree.graph.vp


def test_forest_shares_indices_instead_of_splitting_them():
    forest = Forest([_neuron(1), _neuron(2)])
    paths = forest.get_path(0, 6, parallel=False)
    assert len(paths) == 2
    assert_array_equal(paths[0], [0, 1, 4, 6])
    assert_array_equal(paths[1], [0, 1, 4, 6])
    # Two vertices, two trees. apply() would split this list. Forest must not.
    lcas = forest.get_lowest_common_ancestor([6, 7], parallel=False)
    assert lcas == [4, 4]
    pairs = forest.get_pairwise_path_lengths([[0, 6], [6, 7]], directed=False, parallel=False)
    assert len(pairs) == 2
    assert_allclose(pairs[0], [8.0, 13.0])
    assert_allclose(pairs[1], [8.0, 13.0])


def test_existing_root_distance_binding_is_unchanged():
    tree = _neuron()
    assert tree.get_root_distance(bind=True) is None
    assert "Root_distance" in tree.graph.vp
    assert float(np.array(tree.graph.vp["Root_distance"].a, copy=True)[0]) == 0.0
    tree.get_root_distance(length_type="Euclidean", bind=True)
    assert "Root_euclidean_distance" in tree.graph.vp
    assert not np.allclose(
        tree.graph.vp["Root_distance"].a,
        tree.graph.vp["Root_euclidean_distance"].a,
    )
