"""Tests for geometry_utils.projections."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import projections as proj
from neurosetta.utils.geometry_utils.algebra import magnitude


def test_bisector_xy():
    bx, by, bz = proj.bisector(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    s = np.sqrt(2.0) / 2.0
    assert (bx, by, bz) == pytest.approx((s, s, 0.0))
    assert magnitude(bx, by, bz) == pytest.approx(1.0)


def test_bisector_antiparallel_is_zero():
    assert proj.bisector(1.0, 0.0, 0.0, -1.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0))


def test_bisector_array():
    x1 = np.array([1.0, 1.0])
    y1 = np.zeros(2)
    z1 = np.zeros(2)
    x2 = np.array([0.0, -1.0])
    y2 = np.array([1.0, 0.0])
    z2 = np.zeros(2)
    bx, by, bz = proj.bisector(x1, y1, z1, x2, y2, z2)
    s = np.sqrt(2.0) / 2.0
    assert (bx[0], by[0], bz[0]) == pytest.approx((s, s, 0.0))
    assert (bx[1], by[1], bz[1]) == pytest.approx((0.0, 0.0, 0.0))


def test_reflect_plane_involution():
    nx, ny, nz = 1.0, 0.0, 0.0
    rx, ry, rz = proj.reflect_plane(1.0, 2.0, 3.0, nx, ny, nz)
    assert (rx, ry, rz) == pytest.approx((-1.0, 2.0, 3.0))
    bx, by, bz = proj.reflect_plane(rx, ry, rz, nx, ny, nz)
    assert (bx, by, bz) == pytest.approx((1.0, 2.0, 3.0))


def test_reflect_plane_array():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    z = np.zeros(2)
    xo, yo, zo = proj.reflect_plane(x, y, z, 1.0, 0.0, 0.0)
    assert xo == pytest.approx(np.array([-1.0, 0.0]))
    assert yo == pytest.approx(np.array([0.0, 1.0]))


def test_mirror_axis_involution():
    ax, ay, az = 1.0, 0.0, 0.0
    mx, my, mz = proj.mirror_axis(1.0, 1.0, 0.0, ax, ay, az)
    assert (mx, my, mz) == pytest.approx((1.0, -1.0, 0.0))
    bx, by, bz = proj.mirror_axis(mx, my, mz, ax, ay, az)
    assert (bx, by, bz) == pytest.approx((1.0, 1.0, 0.0))


def test_mirror_axis_array():
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 1.0])
    z = np.zeros(2)
    xo, yo, zo = proj.mirror_axis(x, y, z, 1.0, 0.0, 0.0)
    assert xo == pytest.approx(np.array([1.0, 0.0]))
    assert yo == pytest.approx(np.array([-1.0, -1.0]))
