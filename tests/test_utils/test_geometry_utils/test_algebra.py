"""Tests for geometry_utils.algebra."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import algebra as alg


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


def test_magnitude_zero():
    assert alg.magnitude(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_normalize_array_with_zero():
    x = np.array([3.0, 0.0])
    y = np.array([0.0, 0.0])
    z = np.array([0.0, 0.0])
    nx, ny, nz = alg.normalize(x, y, z)
    assert nx == pytest.approx(np.array([1.0, 0.0]))
    assert ny == pytest.approx(np.array([0.0, 0.0]))
    assert nz == pytest.approx(np.array([0.0, 0.0]))


def test_dot_and_cross_array():
    x1 = np.array([1.0, 0.0])
    y1 = np.array([0.0, 1.0])
    z1 = np.array([0.0, 0.0])
    x2 = np.array([0.0, 1.0])
    y2 = np.array([1.0, 0.0])
    z2 = np.array([0.0, 0.0])
    assert alg.dot(x1, y1, z1, x2, y2, z2) == pytest.approx(np.array([0.0, 0.0]))
    cx, cy, cz = alg.cross(x1, y1, z1, x2, y2, z2)
    assert cz == pytest.approx(np.array([1.0, -1.0]))


def test_cross_parallel_is_zero():
    cx, cy, cz = alg.cross(1.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert (cx, cy, cz) == pytest.approx((0.0, 0.0, 0.0))


def test_scalar_projection_negative():
    assert alg.scalar_projection(1.0, 0.0, 0.0, -1.0, 0.0, 0.0) == pytest.approx(-1.0)


def test_project_reject_zero_axis():
    px, py, pz = alg.project(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    assert (px, py, pz) == pytest.approx((0.0, 0.0, 0.0))
    rx, ry, rz = alg.reject(1.0, 2.0, 3.0, 0.0, 0.0, 0.0)
    assert (rx, ry, rz) == pytest.approx((1.0, 2.0, 3.0))


def test_project_reject_array():
    x = np.array([2.0, 0.0])
    y = np.array([3.0, 4.0])
    z = np.array([0.0, 0.0])
    px, py, pz = alg.project(x, y, z, 1.0, 0.0, 0.0)
    assert px == pytest.approx(np.array([2.0, 0.0]))
    assert py == pytest.approx(np.array([0.0, 0.0]))
    rx, ry, rz = alg.reject(x, y, z, 1.0, 0.0, 0.0)
    assert rx == pytest.approx(np.array([0.0, 0.0]))
    assert ry == pytest.approx(np.array([3.0, 4.0]))


def test_perpendicular_not_normalized():
    px, py, pz = alg.perpendicular(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, normalized=False)
    assert (px, py, pz) == pytest.approx((0.0, 0.0, 6.0))


def test_scale_factor_antiparallel():
    assert alg.scale_factor(2.0, 0.0, 0.0, -4.0, 0.0, 0.0) == pytest.approx(-2.0)


def test_inconsistent_component_shapes_raise():
    with pytest.raises(ValueError, match="inconsistent"):
        alg.magnitude(1.0, np.array([1.0, 2.0]), 0.0)


def test_mismatched_vector_shapes_raise():
    x1 = np.array([1.0, 2.0, 3.0])
    y1 = np.zeros(3)
    z1 = np.zeros(3)
    x2 = np.array([1.0, 2.0])
    y2 = np.zeros(2)
    z2 = np.zeros(2)
    with pytest.raises(ValueError, match="do not match"):
        alg.dot(x1, y1, z1, x2, y2, z2)
