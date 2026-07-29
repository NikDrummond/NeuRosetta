"""Tests for geometry_utils.algebra."""

import numpy as np
import pytest

from NeuRosetta.utils.geometry_utils import algebra as alg


def test_magnitude_scalar():
    assert alg.magnitude(3.0, 4.0, 0.0) == pytest.approx(5.0)


def test_magnitude_array():
    x = np.array([3.0, 0.0])
    y = np.array([4.0, 1.0])
    z = np.array([0.0, 0.0])
    assert alg.magnitude(x, y, z) == pytest.approx(np.array([5.0, 1.0]))


def test_normalize_unit_vector():
    nx, ny, nz = alg.normalize(3.0, 0.0, 0.0)
    assert (nx, ny, nz) == pytest.approx((1.0, 0.0, 0.0))


def test_normalize_zero_vector():
    assert alg.normalize(0.0, 0.0, 0.0) == (0.0, 0.0, 0.0)


def test_dot_orthogonal():
    assert alg.dot(1.0, 0.0, 0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)


def test_cross_right_handed():
    cx, cy, cz = alg.cross(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert (cx, cy, cz) == pytest.approx((0.0, 0.0, 1.0))


def test_scalar_projection():
    assert alg.scalar_projection(2.0, 0.0, 0.0, 1.0, 0.0, 0.0) == pytest.approx(2.0)
    assert np.isnan(alg.scalar_projection(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))


def test_project_and_reject():
    px, py, pz = alg.project(2.0, 3.0, 0.0, 1.0, 0.0, 0.0)
    assert (px, py, pz) == pytest.approx((2.0, 0.0, 0.0))
    rx, ry, rz = alg.reject(2.0, 3.0, 0.0, 1.0, 0.0, 0.0)
    assert (rx, ry, rz) == pytest.approx((0.0, 3.0, 0.0))


def test_perpendicular_normalized():
    px, py, pz = alg.perpendicular(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert alg.magnitude(px, py, pz) == pytest.approx(1.0)


def test_scale_factor_collinear():
    assert alg.scale_factor(2.0, 0.0, 0.0, 4.0, 0.0, 0.0) == pytest.approx(2.0)
    assert np.isnan(alg.scale_factor(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))


def test_broadcast_scalar_against_array():
    x1 = np.array([1.0, 2.0])
    y1 = np.array([0.0, 0.0])
    z1 = np.array([0.0, 0.0])
    result = alg.dot(x1, y1, z1, 1.0, 0.0, 0.0)
    assert result == pytest.approx(np.array([1.0, 2.0]))
