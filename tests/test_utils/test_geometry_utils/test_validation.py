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
