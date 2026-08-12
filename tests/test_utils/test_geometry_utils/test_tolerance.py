"""Tests for geometry_utils.tolerance."""

from neurosetta.utils.geometry_utils import tolerance


def test_almost_zero():
    assert tolerance.almost_zero([0.0, 0.0, 0.0])
    assert not tolerance.almost_zero([1.0, 0.0, 0.0])


def test_almost_unit_length():
    assert tolerance.almost_unit_length([1.0, 0.0, 0.0])
    assert not tolerance.almost_unit_length([2.0, 0.0, 0.0])


def test_almost_collinear():
    assert tolerance.almost_collinear([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    assert not tolerance.almost_collinear([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])


def test_almost_equal():
    assert tolerance.almost_equal([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert not tolerance.almost_equal([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
