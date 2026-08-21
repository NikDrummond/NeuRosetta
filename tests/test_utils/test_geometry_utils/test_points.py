"""Tests for geometry_utils.points."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import points


def test_euclidean_distance_scalar():
    assert points.euclidean_distance(0.0, 0.0, 0.0, 3.0, 4.0, 0.0) == pytest.approx(5.0)


def test_euclidean_distance_array():
    x1 = np.array([0.0, 0.0])
    y1 = np.array([0.0, 0.0])
    z1 = np.array([0.0, 0.0])
    x2 = np.array([3.0, 0.0])
    y2 = np.array([4.0, 1.0])
    z2 = np.array([0.0, 0.0])
    assert points.euclidean_distance(x1, y1, z1, x2, y2, z2) == pytest.approx(np.array([5.0, 1.0]))


def test_argapex_and_apex():
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    z = np.zeros(3)
    assert points.argapex(x, y, z, 1.0, 0.0, 0.0) == 2
    ax, ay, az = points.apex(x, y, z, 1.0, 0.0, 0.0)
    assert (ax, ay, az) == pytest.approx((2.0, 0.0, 0.0))


def test_apex_and_opposite():
    x = np.array([0.0, 1.0, 2.0])
    y = np.zeros(3)
    z = np.zeros(3)
    assert points.apex_and_opposite(x, y, z, 1.0, 0.0, 0.0) == (2, 0)


def test_nearest_and_farthest():
    x = np.array([0.0, 10.0, 5.0])
    y = np.zeros(3)
    z = np.zeros(3)
    assert points.nearest(x, y, z, 4.0, 0.0, 0.0) == 2
    assert points.farthest(x, y, z, 4.0, 0.0, 0.0) == 1


def test_within_radius():
    x = np.array([0.0, 3.0])
    y = np.zeros(2)
    z = np.zeros(2)
    mask = points.within_radius(x, y, z, 0.0, 0.0, 0.0, 2.0)
    assert mask.tolist() == [True, False]


def test_average_unweighted():
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 4.0])
    z = np.array([0.0, 6.0])
    mx, my, mz = points.average(x, y, z)
    assert (mx, my, mz) == pytest.approx((1.0, 2.0, 3.0))


def test_average_weighted():
    x = np.array([0.0, 2.0])
    y = np.zeros(2)
    z = np.zeros(2)
    weights = np.array([1.0, 3.0])
    mx, my, mz, sw = points.average(x, y, z, weights)
    assert mx == pytest.approx(1.5)
    assert sw == pytest.approx(4.0)


def test_euclidean_distance_broadcast():
    x1 = np.array([0.0, 3.0])
    y1 = np.zeros(2)
    z1 = np.zeros(2)
    d = points.euclidean_distance(x1, y1, z1, 0.0, 0.0, 0.0)
    assert d == pytest.approx(np.array([0.0, 3.0]))


def test_argapex_empty():
    empty = np.array([])
    assert points.argapex(empty, empty, empty, 1.0, 0.0, 0.0) == -1
    assert points.apex_and_opposite(empty, empty, empty, 1.0, 0.0, 0.0) == (-1, -1)


def test_within_radius_boundary():
    x = np.array([1.0])
    y = np.zeros(1)
    z = np.zeros(1)
    # strict < (radius + atol)**2; exact radius with atol=0 is outside
    mask = points.within_radius(x, y, z, 0.0, 0.0, 0.0, 1.0, atol=0.0)
    assert mask.tolist() == [False]
    mask = points.within_radius(x, y, z, 0.0, 0.0, 0.0, 1.0, atol=1e-8)
    assert mask.tolist() == [True]


def test_projection_moments_along_x():
    x = np.array([0.0, 2.0, 4.0])
    y = np.zeros(3)
    z = np.zeros(3)
    ax, ay, az = 1.0, 0.0, 0.0
    mean, var, std, pmin, pmax = points.projection_moments(x, y, z, ax, ay, az)
    assert mean == pytest.approx(2.0)
    assert var == pytest.approx(8.0 / 3.0)
    assert std == pytest.approx(np.sqrt(8.0 / 3.0))
    assert pmin == pytest.approx(0.0)
    assert pmax == pytest.approx(4.0)
    assert points.mean_along_axis(x, y, z, ax, ay, az) == pytest.approx(2.0)
    assert points.variance_along_axis(x, y, z, ax, ay, az) == pytest.approx(8.0 / 3.0)
    assert points.std_along_axis(x, y, z, ax, ay, az) == pytest.approx(np.sqrt(8.0 / 3.0))
    assert points.minmax_along_axis(x, y, z, ax, ay, az) == pytest.approx((0.0, 4.0))
    assert points.extent_along_axis(x, y, z, ax, ay, az) == pytest.approx(4.0)
    assert points.rms_along_axis(x, y, z, ax, ay, az) == pytest.approx(np.sqrt(20.0 / 3.0))
    assert points.mean_absolute_along_axis(x, y, z, ax, ay, az) == pytest.approx(2.0)
