"""Tests for geometry_utils.rotations."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import rotations as rot
from neurosetta.utils.geometry_utils.algebra import dot, magnitude


def _apply_axis_angle(v, axis, angle):
    return rot.rotate(*v, *axis, angle, assume_normalized=True)


def test_rotate_90_about_z_array():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    z = np.zeros(2)
    xr, yr, zr = rot.rotate(x, y, z, 0.0, 0.0, 1.0, np.pi / 2)
    assert xr == pytest.approx(np.array([0.0, -1.0]), abs=1e-6)
    assert yr == pytest.approx(np.array([1.0, 0.0]), abs=1e-6)
    assert zr == pytest.approx(np.array([0.0, 0.0]), abs=1e-6)


def test_rotate_identity_and_180():
    assert rot.rotate(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0) == pytest.approx((1.0, 0.0, 0.0))
    xr, yr, zr = rot.rotate(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi)
    assert (xr, yr, zr) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)


def test_rotate_assume_normalized():
    xr, yr, zr = rot.rotate(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2, assume_normalized=True)
    assert (xr, yr, zr) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_rotate_about_point():
    # 90° about z through (1, 0, 0): point at origin goes to (1, -1, 0)
    xr, yr, zr = rot.rotate_about(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2, 1.0, 0.0, 0.0)
    assert (xr, yr, zr) == pytest.approx((1.0, -1.0, 0.0), abs=1e-6)


def test_rotate_about_array_points_scalar_axis():
    x = np.array([0.0, 1.0])
    y = np.zeros(2)
    z = np.zeros(2)
    xr, yr, zr = rot.rotate_about(x, y, z, 0.0, 0.0, 1.0, np.pi / 2, 0.0, 0.0, 0.0)
    assert xr == pytest.approx(np.array([0.0, 0.0]), abs=1e-6)
    assert yr == pytest.approx(np.array([0.0, 1.0]), abs=1e-6)


def test_rotate_about_per_point_axis():
    x = np.array([1.0, 1.0])
    y = np.zeros(2)
    z = np.zeros(2)
    ax = np.array([0.0, 0.0])
    ay = np.array([0.0, 1.0])
    az = np.array([1.0, 0.0])
    xr, yr, zr = rot.rotate_about(x, y, z, ax, ay, az, np.pi / 2, 0.0, 0.0, 0.0)
    # first: 90° about z → (0, 1, 0); second: 90° about y → (0, 0, -1)
    assert (xr[0], yr[0], zr[0]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)
    assert (xr[1], yr[1], zr[1]) == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)


def test_minimum_rotation_already_aligned():
    axis, angle = rot.minimum_rotation_to_align(1.0, 0.0, 0.0, 2.0, 0.0, 0.0)
    assert angle == pytest.approx(0.0)
    assert magnitude(*axis) == pytest.approx(1.0)


def test_minimum_rotation_acute():
    axis, angle = rot.minimum_rotation_to_align(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    assert angle == pytest.approx(np.pi / 2)
    xr, yr, zr = _apply_axis_angle((1.0, 0.0, 0.0), axis, angle)
    assert (xr, yr, zr) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_minimum_rotation_antiparallel():
    axis, angle = rot.minimum_rotation_to_align(1.0, 0.0, 0.0, -1.0, 0.0, 0.0)
    assert angle == pytest.approx(np.pi)
    assert abs(dot(*axis, 1.0, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    xr, yr, zr = _apply_axis_angle((1.0, 0.0, 0.0), axis, angle)
    assert (xr, yr, zr) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)


def test_minimum_rotation_antiparallel_y():
    axis, angle = rot.minimum_rotation_to_align(0.0, 1.0, 0.0, 0.0, -1.0, 0.0)
    assert angle == pytest.approx(np.pi)
    xr, yr, zr = _apply_axis_angle((0.0, 1.0, 0.0), axis, angle)
    assert (xr, yr, zr) == pytest.approx((0.0, -1.0, 0.0), abs=1e-6)


def test_minimum_rotation_assume_normalized():
    axis, angle = rot.minimum_rotation_to_align(
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, assume_normalized=True
    )
    assert angle == pytest.approx(np.pi / 2)


def test_compute_alignment_identity():
    e1, e2, e3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    step1, step2 = rot.compute_alignment_rotation(e1, e2, e3, e1, e2, e3)
    assert step1[1] == pytest.approx(0.0, abs=1e-8)
    assert step2[1] == pytest.approx(0.0, abs=1e-8)
    assert rot.apply_rotation_steps(*e1, step1, step2) == pytest.approx(e1, abs=1e-6)


def test_compute_alignment_and_apply_steps():
    # 90° about z: (x,y,z) -> (y, -x, z)
    e1, e2, e3 = (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)
    b1, b2, b3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    step1, step2 = rot.compute_alignment_rotation(e1, e2, e3, b1, b2, b3)
    r1 = rot.apply_rotation_steps(*e1, step1, step2)
    r2 = rot.apply_rotation_steps(*e2, step1, step2)
    assert r1 == pytest.approx(b1, abs=1e-6)
    assert r2 == pytest.approx(b2, abs=1e-6)


def test_compute_alignment_default_target_axis_aligned():
    e1, e2, e3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    step1, step2 = rot.compute_alignment_rotation(e1, e2, e3)
    r1 = rot.apply_rotation_steps(*e1, step1, step2)
    r2 = rot.apply_rotation_steps(*e2, step1, step2)
    b1 = (0.0, 1.0, 0.0)
    b2 = (1.0, 0.0, 0.0)
    assert abs(dot(*r1, *b1)) == pytest.approx(1.0, abs=1e-6)
    assert abs(dot(*r2, *b2)) == pytest.approx(1.0, abs=1e-6)


def test_compute_alignment_flips_opposite_e1():
    e1, e2, e3 = (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    b1, b2, b3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    step1, step2 = rot.compute_alignment_rotation(e1, e2, e3, b1, b2, b3)
    # after sign repair, e1 maps onto b1 with (near) identity
    r1 = rot.apply_rotation_steps(1.0, 0.0, 0.0, step1, step2)
    assert r1 == pytest.approx(b1, abs=1e-6)


def test_compute_alignment_repairs_handedness():
    e1, e2, e3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0)
    b1, b2, b3 = (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    step1, step2 = rot.compute_alignment_rotation(e1, e2, e3, b1, b2, b3)
    r1 = rot.apply_rotation_steps(*e1, step1, step2)
    r2 = rot.apply_rotation_steps(*e2, step1, step2)
    assert r1 == pytest.approx(b1, abs=1e-6)
    assert r2 == pytest.approx(b2, abs=1e-6)


def test_apply_rotation_steps_identity():
    step = ((0.0, 0.0, 1.0), 0.0)
    assert rot.apply_rotation_steps(1.0, 2.0, 3.0, step, step) == pytest.approx((1.0, 2.0, 3.0))
