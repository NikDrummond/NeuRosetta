"""Tests for geometry_utils._validation."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import _validation as val


def test_pluralize():
    assert val.pluralize("dimension", 1) == "dimension"
    assert val.pluralize("dimension", 2) == "dimensions"


def test_check_value_exact_shape():
    arr = np.zeros((3, 4))
    assert val.check_value(arr, (3, 4)) is None


def test_check_value_wildcard():
    arr = np.zeros((5, 3))
    assert val.check_value(arr, (-1, 3)) == 5


def test_check_value_raises_on_mismatch():
    with pytest.raises(ValueError, match="shape"):
        val.check_value(np.zeros((2, 3)), (3, 2), name="coords")


def test_check_value_any_accepts_matching_shape():
    arr = np.zeros((4, 3))
    assert val.check_value_any(arr, (-1, 3), (3, -1)) == 4


def test_check_from_locals():
    x = np.arange(5.0)
    val.check(locals(), "x", (-1,))


def test_columnize_already_2d():
    arr = np.arange(6.0).reshape(2, 3)
    out, was_col, _restore = val.columnize(arr, shape=(-1, 3))
    assert out.shape == (2, 3)
    assert was_col is True


def test_columnize_from_1d_length_3():
    arr = np.array([1.0, 2.0, 3.0])
    out, was_col, restore = val.columnize(arr, shape=(-1, 3))
    assert out.shape == (-1, 3) or out.shape == (1, 3)
    assert was_col is False


def test_raise_dimension_error_message():
    with pytest.raises(ValueError, match="Not sure what to do"):
        val.raise_dimension_error(np.zeros((2, 3, 4)))


def test_raise_dimension_error_two_inputs():
    with pytest.raises(ValueError, match="dimension"):
        val.raise_dimension_error(np.zeros((2,)), np.zeros((3, 4)))


def test_check_value_none_and_non_array():
    with pytest.raises(ValueError, match="None"):
        val.check_value(None, (3,), name="coords")
    with pytest.raises(ValueError, match="list"):
        val.check_value([1.0, 2.0, 3.0], (3,), name="coords")


def test_check_value_any_no_shapes_and_mismatch():
    with pytest.raises(ValueError, match="At least one shape"):
        val.check_value_any(np.zeros((3,)))
    with pytest.raises(ValueError, match="shape"):
        val.check_value_any(np.zeros((2, 2)), (-1, 3), (3,), name="coords")


def test_columnize_restore():
    arr = np.array([1.0, 2.0, 3.0])
    out, was_col, restore = val.columnize(arr, shape=(-1, 3))
    assert was_col is False
    assert restore(out).shape == (3,)


def test_columnize_invalid_shape():
    arr = np.zeros((2, 3))
    with pytest.raises(ValueError, match="tuple"):
        val.columnize(arr, shape=[-1, 3])
    with pytest.raises(ValueError, match="two dimensions"):
        val.columnize(arr, shape=(-1,))


def test_check_scalar_inconsistent_shapes():
    with pytest.raises(ValueError, match="inconsistent"):
        val._check_scalar(1.0, np.array([1.0]), 0.0)


def test_check_vector_broadcast_and_broadcast_three():
    x = np.array([1.0, 2.0])
    y = np.zeros(2)
    z = np.zeros(2)
    assert val._check_vector_broadcast(x, y, z, 0.0, 1.0, 0.0) is True
    assert val._check_vector_broadcast(x, y, z, x, y, z) is False
    bx = val._broadcast_vectors(x, y, z, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    assert len(bx) == 9
    assert np.allclose(bx[3], 0.0)
    assert np.allclose(bx[4], 1.0)
    assert np.allclose(bx[8], 1.0)
