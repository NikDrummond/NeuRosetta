"""Tests for geometry_utils.basis."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils.basis import (
    basis,
    change_basis_scalar,
    change_basis_xyz,
    decompose_basis_scalar,
    decompose_basis_xyz,
    inverse_change_basis_xyz,
    reconstruct_basis_scalar,
    reconstruct_basis_xyz,
)


def test_basis_vectors():
    assert np.allclose(basis.x, [1.0, 0.0, 0.0])
    assert np.allclose(basis.y, [0.0, 1.0, 0.0])
    assert np.allclose(basis.z, [0.0, 0.0, 1.0])
    assert np.allclose(basis.neg_x, [-1.0, 0.0, 0.0])


def test_basis_vectors_readonly():
    with pytest.raises(ValueError):
        basis.x[0] = 5.0


def test_negative_basis_vectors():
    assert np.allclose(basis.neg_y, [0.0, -1.0, 0.0])
    assert np.allclose(basis.neg_z, [0.0, 0.0, -1.0])


def test_decompose_reconstruct_roundtrip_scalar():
    b1, b2, b3 = basis.x, basis.y, basis.z
    c1, c2, c3 = decompose_basis_scalar(3.0, 4.0, 5.0, *b1, *b2, *b3)
    assert (c1, c2, c3) == pytest.approx((3.0, 4.0, 5.0))
    x, y, z = reconstruct_basis_scalar(c1, c2, c3, *b1, *b2, *b3)
    assert (x, y, z) == pytest.approx((3.0, 4.0, 5.0))


def test_decompose_reconstruct_roundtrip_array():
    b1, b2, b3 = basis.x, basis.y, basis.z
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 2.0])
    z = np.array([0.0, 3.0])
    c1, c2, c3 = decompose_basis_xyz(x, y, z, *b1, *b2, *b3)
    xr, yr, zr = reconstruct_basis_xyz(c1, c2, c3, *b1, *b2, *b3)
    assert xr == pytest.approx(x)
    assert yr == pytest.approx(y)
    assert zr == pytest.approx(z)


def test_change_basis_aliases_decompose():
    b1, b2, b3 = basis.x, basis.y, basis.z
    assert change_basis_scalar(1.0, 2.0, 3.0, *b1, *b2, *b3) == pytest.approx((1.0, 2.0, 3.0))
    x = np.array([1.0, 4.0])
    y = np.array([2.0, 5.0])
    z = np.array([3.0, 6.0])
    c1, c2, c3 = change_basis_xyz(x, y, z, *b1, *b2, *b3)
    xr, yr, zr = inverse_change_basis_xyz(c1, c2, c3, *b1, *b2, *b3)
    assert xr == pytest.approx(x)
    assert yr == pytest.approx(y)
    assert zr == pytest.approx(z)
