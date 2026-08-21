"""Tests for the generic vedo actor helpers."""

from __future__ import annotations

import numpy as np
import pytest
from vedo import Sphere

from neurosetta.utils.vedo_utils import (
    combined_bounds,
    get_marker_size,
    make_assembly,
    make_lines,
    make_point_marker,
    random_colour,
    resolve_colour,
    set_actor_alpha,
    set_actor_colour,
    set_marker_size,
)


def test_random_colour_is_plain_ints_in_range():
    colour = random_colour(seed=0)
    assert len(colour) == 3
    assert all(isinstance(v, int) and 0 <= v <= 255 for v in colour)


def test_random_colour_is_reproducible_with_a_seed():
    assert random_colour(seed=7) == random_colour(seed=7)


def test_resolve_colour_normalises_specifiers():
    assert resolve_colour("k") == (0.0, 0.0, 0.0)
    assert resolve_colour([255, 0, 0]) == pytest.approx((1.0, 0.0, 0.0))


def test_make_lines_builds_one_cell_per_pair():
    starts = np.zeros((4, 3))
    stops = np.ones((4, 3))
    lines = make_lines(starts, stops, c="k", lw=2)
    assert lines.ncells == 4
    assert lines.lw() == 2


def test_make_point_marker_accepts_nested_coordinates():
    marker = make_point_marker([[1.0, 2.0, 3.0]], r=6)
    assert not isinstance(marker, Sphere)
    assert get_marker_size(marker) == pytest.approx(6)


def test_make_point_marker_as_sphere():
    marker = make_point_marker([0.0, 0.0, 0.0], as_sphere=True, r=4)
    assert isinstance(marker, Sphere)
    assert get_marker_size(marker) == pytest.approx(4)


@pytest.mark.parametrize("as_sphere", [False, True])
def test_set_marker_size_round_trips(as_sphere):
    marker = make_point_marker([0.0, 0.0, 0.0], as_sphere=as_sphere, r=5)
    set_marker_size(marker, 11)
    assert get_marker_size(marker) == pytest.approx(11)


def test_set_actor_helpers_ignore_none():
    set_actor_colour(None, "red")
    set_actor_alpha(None, 0.5)

    lines = make_lines(np.zeros((2, 3)), np.ones((2, 3)))
    set_actor_colour(lines, None)
    set_actor_alpha(lines, None)
    set_actor_colour(lines, "red")
    set_actor_alpha(lines, 0.5)
    assert np.allclose(lines.color(), resolve_colour("red"))
    assert lines.alpha() == pytest.approx(0.5)


def test_combined_bounds_spans_all_actors():
    a = make_lines(np.zeros((1, 3)), np.ones((1, 3)))
    b = make_lines(np.full((1, 3), 2.0), np.full((1, 3), 3.0))
    xmin, xmax, ymin, ymax, zmin, zmax = combined_bounds([a, b, None])
    assert (xmin, ymin, zmin) == pytest.approx((0.0, 0.0, 0.0))
    assert (xmax, ymax, zmax) == pytest.approx((3.0, 3.0, 3.0))


def test_combined_bounds_of_nothing_is_none():
    assert combined_bounds([None]) is None


def test_make_assembly_drops_none():
    lines = make_lines(np.zeros((1, 3)), np.ones((1, 3)))
    assert len(make_assembly([lines, None]).unpack()) == 1
