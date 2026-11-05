"""Tree and Forest bindings for subtree metrics, Strahler order, and Sholl."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import neurosetta as nr
from neurosetta.ops.tree_graphs.tree_subtrees import get_subtree_scores
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


def test_historical_and_new_imports():
    assert nr.get_subtree_scores is get_subtree_scores
    tree = make_synthetic_tree(12, units="micron")
    assert tree.get_strahler_order().shape == (tree.count_nodes(),)
    assert tree.get_max_strahler_order() >= 1
    assert tree.get_subtree_vertex_count()[tree.get_root_index()] == tree.count_nodes()
    leaves = tree.get_subtree_leaf_count()
    edges = tree.get_subtree_edge_count()
    verts = tree.get_subtree_vertex_count()
    assert int(leaves[tree.get_leaf_indices()].min()) == 1
    assert np.array_equal(verts, edges + 1)
    cable = tree.get_subtree_path_length()
    assert_allclose(cable[tree.get_root_index()], tree.get_total_cable_length())


def test_reduced_path_length_is_required():
    reduced = make_synthetic_tree(20, seed=1, units="micron").copy()
    reduced.get_reduced_tree(inplace=True)
    del reduced.graph.ep["Path_length"]
    with pytest.raises(ValueError, match="Path_length"):
        reduced.get_subtree_path_length()
    with pytest.raises(ValueError, match="Path_length"):
        reduced.get_sholl_profile([1.0, 2.0], mode="path_length")


def test_forest_batch_and_shared_radii():
    forest = make_synthetic_forest(n_trees=2, n=15, seed=2, units="micron")
    orders = forest.get_max_strahler_order(parallel=False)
    assert len(orders) == 2
    # Length equals the forest size. apply() must not split this array.
    profiles = forest.get_sholl_profile([1.0, 2.0], mode="radial", parallel=False)
    assert len(profiles) == 2
    assert profiles[0].intersections.shape == (2,)
    assert profiles[0].units is not None
    assert profiles[1].radii.tolist() == [1.0, 2.0]


def test_sholl_derived_on_tree_profile():
    tree = make_synthetic_tree(18, seed=4, units="micron")
    profile = tree.get_sholl_profile([5.0, 15.0, 40.0], mode="euclidean_length")
    assert nr.get_sholl_peak_count(profile) == int(profile.intersections.max())
    assert nr.get_sholl_peak_radius(profile) in set(profile.radii.tolist())
    assert nr.get_sholl_area(profile) >= 0
