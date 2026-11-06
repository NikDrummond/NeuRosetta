"""Hand-calculated measurements on the reference morphologies.

Integer topology is exact. Lengths here are integers or halves, which are
exact in float64, so absolute tolerance 1e-12 is enough and is far below a
biologically meaningful length error at these magnitudes. Angles of
axis-aligned edges are exact multiples of π/2; the same absolute tolerance
is used in radians.
"""

from __future__ import annotations

import numpy as np
import pytest
from graph_tool.all import Graph
from numpy.testing import assert_allclose

import neurosetta as nr
from neurosetta.testing import (
    asymmetric_branch,
    irregular_curve,
    multifurcation,
    nonuniform_chain,
    simple_bifurcation,
    single_node,
    spatial_tetrahedron,
    straight_chain,
    symmetric_binary,
    zero_length_edge,
)
from neurosetta.utils.graph_utils.tree_validation import validate_rooted_tree

pytestmark = pytest.mark.filterwarnings("ignore:.*units.*:UserWarning")

_ATOL = 1e-12


def test_single_node_counts_and_strahler():
    tree = single_node()
    assert tree.count_nodes() == nr.count_nodes(tree) == 1
    assert tree.count_edges() == 0
    assert tree.count_leaves() == 1
    assert tree.count_branches() == 0
    assert tree.get_total_cable_length() == 0.0
    assert tree.get_strahler_order().tolist() == [1]
    assert tree.get_max_strahler_order() == 1


def test_straight_chain_lengths_depth_and_sholl():
    tree = straight_chain()
    assert tree.count_nodes() == 4
    assert tree.count_edges() == 3
    assert tree.count_sections() == 1
    assert tree.count_transitive_nodes() == 2
    assert tree.get_max_depth() == 3
    assert_allclose(tree.get_edge_length(bind=False), [1, 1, 1], atol=_ATOL)
    assert tree.get_total_cable_length() == pytest.approx(3.0, abs=_ATOL)
    assert_allclose(tree.get_root_distance(bind=False), [0, 1, 2, 3], atol=_ATOL)
    assert_allclose(tree.get_subtree_path_length(), [3, 2, 1, 0], atol=_ATOL)
    assert tree.get_strahler_order().tolist() == [1, 1, 1, 1]
    profile = tree.get_sholl_profile([0.0, 0.5, 1.5, 2.5, 3.5], mode="path_length")
    assert profile.intersections.tolist() == [0, 1, 1, 1, 0]


def test_nonuniform_chain_cable_is_not_node_spacing_in_index():
    tree = nonuniform_chain()
    assert_allclose(tree.get_edge_length(bind=False), [1, 2, 4], atol=_ATOL)
    assert tree.get_total_cable_length() == pytest.approx(7.0, abs=_ATOL)
    assert_allclose(tree.get_root_distance(bind=False), [0, 1, 3, 7], atol=_ATOL)
    profile = tree.get_sholl_profile([0.5, 2.0, 5.0], mode="path_length")
    assert profile.intersections.tolist() == [1, 1, 1]


def test_simple_bifurcation_angle_is_a_right_angle():
    tree = simple_bifurcation()
    assert tree.count_bifurcations(include_root=False) == 1
    assert tree.count_branches() == 1
    assert tree.count_sections() == 3
    _pc1, _pc2, child = tree.get_bifurcation_angles(degrees=False)
    assert_allclose(child, [np.pi / 2], atol=_ATOL)
    parent_angles = np.sort(np.concatenate([_pc1, _pc2]))
    assert_allclose(parent_angles, [np.pi / 2, np.pi], atol=_ATOL)


def test_symmetric_binary_strahler_and_cable():
    tree = symmetric_binary()
    assert tree.count_leaves() == 4
    assert tree.count_branches() == 3
    assert tree.count_bifurcations(include_root=False) == 2
    assert tree.count_bifurcations(include_root=True) == 3
    assert tree.get_total_cable_length() == pytest.approx(6.0, abs=_ATOL)
    order = tree.get_strahler_order()
    assert int(order[0]) == 3
    assert order[1:3].tolist() == [2, 2]
    assert order[3:].tolist() == [1, 1, 1, 1]
    assert tree.get_max_strahler_order() == 3


def test_asymmetric_partition_and_strahler():
    tree = asymmetric_branch()
    # This tree has no degree-2 vertices, so check_reduced() is already true
    # and get_edge_length stores Euclidean_length. Cable modes require
    # Path_length and do not read the Euclidean property. On an unreduced
    # arbor those two lengths are the same, so copy them across.
    lengths = np.array(tree.get_edge_length(bind=False), dtype=float, copy=True)
    tree.graph.ep["Path_length"] = tree.graph.new_ep("double", lengths)
    assert tree.get_strahler_order().tolist() == [2, 1, 2, 1, 1]
    unweighted = tree.get_partition_asymmetry(mode="unweighted", bind=False)
    van_pelt = tree.get_partition_asymmetry(mode="van_pelt", bind=False)
    cable = tree.get_partition_asymmetry(mode="cable", bind=False)
    assert_allclose(unweighted, [0.5, 0, 0, 0, 0], atol=_ATOL)
    assert_allclose(van_pelt, [1, 0, 0, 0, 0], atol=_ATOL)
    assert_allclose(cable, [0.5, 0, 0, 0, 0], atol=_ATOL)
    # Parent edge is excluded: vertex 2 owns the two length-1 daughter edges.
    assert_allclose(tree.get_subtree_path_length(), [4, 0, 2, 0, 0], atol=_ATOL)


def test_multifurcation_is_a_branch_but_not_a_bifurcation():
    tree = multifurcation()
    assert tree.count_branches() == 1
    assert tree.count_bifurcations(include_root=True) == 0
    assert tree.count_sections() == 3
    assert tree.get_strahler_order().tolist() == [2, 1, 1, 1]
    assert tree.get_convex_hull_volume() == pytest.approx(1.0 / 6.0, rel=1e-9, abs=1e-12)


def test_zero_length_edge_counts_and_does_not_add_cable():
    tree = zero_length_edge()
    assert tree.count_edges() == 2
    assert_allclose(tree.get_edge_length(bind=False), [0, 1], atol=_ATOL)
    assert tree.get_total_cable_length() == pytest.approx(1.0, abs=_ATOL)


def test_irregular_curve_reduction_keeps_cable_not_the_chord():
    tree = irregular_curve()
    assert tree.get_total_cable_length() == pytest.approx(3.5, abs=_ATOL)
    profile = tree.get_sholl_profile([0.5, 1.5, 2.5, 3.5], mode="path_length")
    assert profile.intersections.tolist() == [1, 1, 1, 1]
    tree.get_reduced_tree(inplace=True)
    assert tree.count_edges() == 1
    assert float(tree.graph.ep["Path_length"].a[0]) == pytest.approx(3.5, abs=_ATOL)
    ends = tree.get_node_coordinates()
    chord = float(np.linalg.norm(ends[-1] - ends[0]))
    assert chord == pytest.approx(np.sqrt(5.25), abs=_ATOL)
    assert chord != pytest.approx(3.5, abs=1e-6)
    # Default edge length on the reduced tree is the stored cable.
    assert float(tree.get_edge_length(bind=False)[0]) == pytest.approx(3.5, abs=_ATOL)
    euclid = tree.get_edge_length(bind=False, recalculate=True)
    assert float(euclid[0]) == pytest.approx(chord, abs=_ATOL)


def test_tetrahedron_volume():
    tree = spatial_tetrahedron()
    assert tree.get_convex_hull_volume() == pytest.approx(1.0 / 6.0, rel=1e-9, abs=1e-12)


def test_nan_coordinate_does_not_become_a_finite_length():
    tree = straight_chain()
    del tree.graph.ep["Path_length"]
    tree.graph.vp["x"][1] = np.nan
    length = tree.get_total_cable_length()
    assert np.isnan(length)


def test_cycle_is_rejected():
    graph = Graph(directed=True)
    graph.add_vertex(3)
    graph.add_edge_list([(0, 1), (1, 2), (2, 1)])
    with pytest.raises(ValueError, match="cycle|root|parent"):
        validate_rooted_tree(graph)


def test_two_roots_are_rejected():
    graph = Graph(directed=True)
    graph.add_vertex(2)
    with pytest.raises(ValueError, match="root"):
        validate_rooted_tree(graph)
