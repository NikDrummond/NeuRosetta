"""Tests for geometry_utils.angles."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import angles, rotate


def test_angle_right_angle_degrees():
    assert angles.angle(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, degrees=True) == pytest.approx(90.0)


def test_angle_parallel_radians():
    assert angles.angle(1.0, 0.0, 0.0, 2.0, 0.0, 0.0, degrees=False) == pytest.approx(0.0)


def test_angle_zero_vector_returns_nan():
    assert np.isnan(angles.angle(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))


def test_planar_angle_in_xy_plane():
    result = angles.planar_angle(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, degrees=True)
    assert result == pytest.approx(90.0)


def test_signed_angle_sign():
    pos = angles.signed_angle(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, degrees=True)
    neg = angles.signed_angle(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, degrees=True)
    assert pos == pytest.approx(90.0)
    assert neg == pytest.approx(-90.0)


def test_rotate_90_about_z():
    xr, yr, zr = rotate(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, np.pi / 2)
    assert (xr, yr, zr) == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_align_with_flips_when_opposite():
    x, y, z = angles.align_with(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0))


def test_align_with_reverse():
    x, y, z = angles.align_with(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, reverse=True)
    assert (x, y, z) == pytest.approx((-1.0, 0.0, 0.0))


def test_angle_array():
    x1 = np.array([1.0, 1.0])
    y1 = np.zeros(2)
    z1 = np.zeros(2)
    x2 = np.zeros(2)
    y2 = np.array([1.0, -1.0])
    z2 = np.zeros(2)
    result = angles.angle(x1, y1, z1, x2, y2, z2, degrees=True)
    assert result == pytest.approx(np.array([90.0, 90.0]))


def test_angle_assume_normalized():
    assert angles.angle(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, assume_normalized=True) == pytest.approx(
        np.pi / 2
    )


def test_angle_antiparallel():
    assert angles.angle(1.0, 0.0, 0.0, -1.0, 0.0, 0.0, degrees=True) == pytest.approx(180.0)


def test_planar_and_signed_angle_array():
    x1 = np.array([1.0, 1.0])
    y1 = np.zeros(2)
    z1 = np.zeros(2)
    x2 = np.zeros(2)
    y2 = np.array([1.0, -1.0])
    z2 = np.zeros(2)
    look = (0.0, 0.0, 1.0)
    planar = angles.planar_angle(x1, y1, z1, x2, y2, z2, *look, degrees=True)
    signed = angles.signed_angle(x1, y1, z1, x2, y2, z2, *look, degrees=True)
    assert planar == pytest.approx(np.array([90.0, 90.0]))
    assert signed == pytest.approx(np.array([90.0, -90.0]))


def test_signed_angle_zero_and_180():
    z_look = (0.0, 0.0, 1.0)
    assert angles.signed_angle(
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, *z_look, degrees=True
    ) == pytest.approx(0.0)
    assert angles.signed_angle(
        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, *z_look, degrees=True
    ) == pytest.approx(180.0)


def test_align_with_already_aligned():
    x, y, z = angles.align_with(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    assert (x, y, z) == pytest.approx((1.0, 0.0, 0.0))


def test_align_with_array():
    x = np.array([1.0, -1.0])
    y = np.zeros(2)
    z = np.zeros(2)
    xo, yo, zo = angles.align_with(x, y, z, 1.0, 0.0, 0.0)
    assert xo == pytest.approx(np.array([1.0, 1.0]))


def test_angular_mean_wraparound():
    a = np.deg2rad(np.array([350.0, 10.0]))
    mean = angles.angular_mean(a)
    assert np.cos(mean) == pytest.approx(1.0, abs=1e-6)
    assert np.sin(mean) == pytest.approx(0.0, abs=1e-6)


def test_angular_mean_weighted():
    a = np.array([0.0, np.pi])
    w = np.array([1.0, 0.0])
    assert angles.angular_mean(a, w) == pytest.approx(0.0, abs=1e-6)


def test_angular_var_identical_and_zero_weight():
    a = np.zeros(4)
    assert angles.angular_var(a) == pytest.approx(0.0)
    assert np.isnan(angles.angular_var(a, np.zeros(4)))
