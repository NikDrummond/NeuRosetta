"""Sholl intersections for radial, path-length, and Euclidean-length modes."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from tree_builders import make_directed_tree

from neurosetta.utils.graph_utils import (
    ShollProfile,
    sholl_area,
    sholl_peak_count,
    sholl_peak_radius,
    sholl_profile,
)
from neurosetta.utils.graph_utils.graph_editing import reduce_graph
from neurosetta.utils.graph_utils.node_types import infer_node_types


def _line(length=3):
    coords = [(i, 0.0, 0.0) for i in range(length + 1)]
    edges = [(i, i + 1) for i in range(length)]
    return make_directed_tree(
        edges,
        coords=coords,
        path_length=np.ones(length),
        euclidean=np.ones(length),
    )


def test_straight_line_radial_matches_path():
    g = _line(3)
    radii = [0.5, 1.5, 2.5]
    radial = sholl_profile(g, radii, mode="radial")
    path = sholl_profile(g, radii, mode="path_length")
    assert_array_equal(radial.intersections, [1, 1, 1])
    assert_array_equal(path.intersections, [1, 1, 1])
    assert radial.mode == "radial"
    assert_allclose(radial.center, [0, 0, 0])
    assert path.center is None
    assert radial.units is None


def test_bent_tree_separates_radial_and_path():
    g = make_directed_tree(
        [(0, 1), (1, 2)],
        coords=[(0, 0, 0), (1, 0, 0), (1, 1, 0)],
        path_length=[1.0, 1.0],
        euclidean=[1.0, 1.0],
    )
    radial = sholl_profile(g, [1.7], mode="radial")
    path = sholl_profile(g, [1.7], mode="path_length")
    assert int(radial.intersections[0]) == 0
    assert int(path.intersections[0]) == 1


def test_path_and_euclidean_diverge_when_weights_differ():
    g = make_directed_tree(
        [(0, 1)],
        coords=[(0, 0, 0), (1, 0, 0)],
        path_length=[5.0],
        euclidean=[1.0],
    )
    path = sholl_profile(g, [2.0], mode="path_length")
    eucl = sholl_profile(g, [2.0], mode="euclidean_length")
    assert int(path.intersections[0]) == 1
    assert int(eucl.intersections[0]) == 0


def test_segment_through_sphere_counts_two():
    g = make_directed_tree(
        [(0, 1)],
        coords=[(-2.0, 0.5, 0.0), (2.0, 0.5, 0.0)],
    )
    profile = sholl_profile(g, [1.0], mode="radial", center=(0.0, 0.0, 0.0))
    assert int(profile.intersections[0]) == 2


def test_tangency_is_not_counted():
    g = make_directed_tree(
        [(0, 1)],
        coords=[(-2.0, 1.0, 0.0), (2.0, 1.0, 0.0)],
    )
    profile = sholl_profile(g, [1.0], mode="radial", center=(0.0, 0.0, 0.0))
    assert int(profile.intersections[0]) == 0


def test_vertex_and_branch_on_shell_counted_once():
    line = make_directed_tree([(0, 1)], coords=[(0, 0, 0), (1, 0, 0)])
    on_vertex = sholl_profile(line, [1.0], mode="radial")
    assert int(on_vertex.intersections[0]) == 1

    branch = make_directed_tree(
        [(0, 1), (1, 2), (1, 3)],
        coords=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (1, 1, 0)],
    )
    on_branch = sholl_profile(branch, [1.0], mode="radial")
    assert int(on_branch.intersections[0]) == 1


def test_long_edge_crosses_many_radii_and_nonuniform_area():
    g = make_directed_tree([(0, 1)], coords=[(0, 0, 0), (10, 0, 0)], path_length=[10.0])
    radii = np.arange(1, 10)
    profile = sholl_profile(g, radii, mode="path_length")
    assert_array_equal(profile.intersections, np.ones(9, dtype=int))
    assert sholl_peak_count(profile) == 1
    assert sholl_peak_radius(profile) == 1.0

    tied = ShollProfile(
        radii=np.array([0.0, 1.0, 3.0]),
        intersections=np.array([0, 2, 2]),
        mode="path_length",
    )
    assert sholl_peak_radius(tied) == 1.0
    assert sholl_peak_count(tied) == 2
    # Trapezoid: (0+2)/2 * 1 + (2+2)/2 * 2 = 5.
    assert_allclose(sholl_area(tied), 5.0)
    assert sholl_area(ShollProfile(radii=[1.0], intersections=[4], mode="radial")) == 0.0


def test_radius_zero_and_invalid_radii():
    g = _line(1)
    zero = sholl_profile(g, [0.0, 0.5], mode="path_length")
    assert int(zero.intersections[0]) == 0
    assert int(zero.intersections[1]) == 1
    with pytest.raises(ValueError):
        sholl_profile(g, [1.0, 1.0], mode="radial")
    with pytest.raises(ValueError):
        sholl_profile(g, [2.0, 1.0], mode="radial")
    with pytest.raises(ValueError):
        sholl_profile(g, [-1.0, 1.0], mode="radial")
    with pytest.raises(ValueError):
        sholl_profile(g, [[1.0, 2.0]], mode="radial")
    with pytest.raises(ValueError):
        sholl_profile(g, [np.nan], mode="radial")


def test_center_rejected_for_path_modes_and_missing_properties():
    g = _line(1)
    with pytest.raises(ValueError, match="center"):
        sholl_profile(g, [0.5], mode="path_length", center=(0, 0, 0))
    bare = make_directed_tree([(0, 1)], coords=[(0, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError, match="Path_length"):
        sholl_profile(bare, [0.5], mode="path_length")
    with pytest.raises(ValueError, match="Euclidean_length"):
        sholl_profile(bare, [0.5], mode="euclidean_length")
    no_xyz = make_directed_tree([(0, 1)], path_length=[1.0])
    with pytest.raises(ValueError, match="x"):
        sholl_profile(no_xyz, [0.5], mode="radial")


def test_sholl_does_not_write_root_distance():
    g = _line(2)
    before = set(g.vp.keys())
    sholl_profile(g, [0.5, 1.5], mode="path_length")
    sholl_profile(g, [0.5, 1.5], mode="euclidean_length")
    assert set(g.vp.keys()) == before


def test_reduced_radial_uses_chord_and_path_uses_cable():
    coords = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)]
    g = make_directed_tree(
        [(0, 1), (1, 2), (2, 3)],
        coords=coords,
        path_length=[1.0, 1.0, 1.0],
    )
    g.vp["radius"] = g.new_vp("double", np.ones(4))
    g.vp["node_type"] = g.new_vp("int", infer_node_types(g))
    g.gp["ID"] = g.new_gp("long", 3)
    g.gp["metadata"] = g.new_gp("object", {"isReduced": True, "file_path": ""})
    # Mark reduced only after we still have the detailed graph for comparison.
    detailed = sholl_profile(g, [0.5, 1.5, 2.5], mode="path_length")
    g.gp["metadata"] = g.new_gp("object", {"isReduced": False, "file_path": ""})
    reduced = reduce_graph(g)
    radii = [0.5, 1.5, 2.5]
    path = sholl_profile(reduced, radii, mode="path_length")
    radial = sholl_profile(reduced, radii, mode="radial")
    assert_array_equal(path.intersections, detailed.intersections)
    assert_array_equal(path.intersections, [1, 1, 1])
    # Chord length is sqrt(3) ≈ 1.732, so radius 2.5 is not reached.
    assert_array_equal(radial.intersections, [1, 1, 0])
