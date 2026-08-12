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
