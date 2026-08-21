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


def test_almost_collinear_antiparallel():
    assert tolerance.almost_collinear([1.0, 0.0, 0.0], [-2.0, 0.0, 0.0])


def test_almost_zero_custom_atol():
    assert tolerance.almost_zero([1e-6, 0.0, 0.0], atol=1e-5)
    assert not tolerance.almost_zero([1e-6, 0.0, 0.0], atol=1e-8)


def test_almost_unit_length_near_one():
    assert tolerance.almost_unit_length([1.0 + 1e-9, 0.0, 0.0], atol=1e-8)
