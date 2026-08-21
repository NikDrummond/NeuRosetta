"""Tests for geometry_utils.transforms."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import transforms as tf


def test_translate_scalar_and_array():
    assert tf.translate(1.0, 2.0, 3.0, 1.0, -1.0, 0.0) == pytest.approx((2.0, 1.0, 3.0))
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0])
    z = np.array([0.0, 3.0])
    xo, yo, zo = tf.translate(x, y, z, 1.0, 1.0, 1.0)
    assert xo == pytest.approx(np.array([1.0, 2.0]))
    assert yo == pytest.approx(np.array([1.0, 3.0]))
    assert zo == pytest.approx(np.array([1.0, 4.0]))


def test_recenter():
    assert tf.recenter(5.0, 1.0, 0.0, 5.0, 0.0, 0.0) == pytest.approx((0.0, 1.0, 0.0))
    x = np.array([5.0, 6.0])
    y = np.zeros(2)
    z = np.zeros(2)
    xo, yo, zo = tf.recenter(x, y, z, 5.0, 0.0, 0.0)
    assert xo == pytest.approx(np.array([0.0, 1.0]))


def test_scale_about_identity_zero_and_negative():
    assert tf.scale_about(2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0) == pytest.approx(
        (2.0, 0.0, 0.0)
    )
    assert tf.scale_about(2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert tf.scale_about(2.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0) == pytest.approx(
        (-2.0, 0.0, 0.0)
    )


def test_scale_about_array():
    x = np.array([2.0, 0.0])
    y = np.zeros(2)
    z = np.zeros(2)
    xo, yo, zo = tf.scale_about(x, y, z, 2.0, 1.0, 1.0, 0.0, 0.0, 0.0)
    assert xo == pytest.approx(np.array([4.0, 0.0]))


def test_center_at_centroid_roundtrip():
    x = np.array([0.0, 2.0])
    y = np.array([0.0, 4.0])
    z = np.array([0.0, 6.0])
    xo, yo, zo, cx, cy, cz = tf.center_at_centroid(x, y, z)
    assert (cx, cy, cz) == pytest.approx((1.0, 2.0, 3.0))
    assert xo == pytest.approx(np.array([-1.0, 1.0]))
    assert yo == pytest.approx(np.array([-2.0, 2.0]))
    assert zo == pytest.approx(np.array([-3.0, 3.0]))
    assert xo + cx == pytest.approx(x)
    assert yo + cy == pytest.approx(y)
    assert zo + cz == pytest.approx(z)
