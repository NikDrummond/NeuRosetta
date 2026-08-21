"""Tests for geometry_utils.scaling."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils.basis import basis
from neurosetta.utils.geometry_utils.scaling import scale_along_basis


def test_scale_along_basis_anisotropic():
    x = np.array([1.0])
    y = np.array([2.0])
    z = np.array([3.0])
    xo, yo, zo = scale_along_basis(x, y, z, (2.0, 3.0, 4.0), basis.x, basis.y, basis.z)
    assert (xo[0], yo[0], zo[0]) == pytest.approx((2.0, 6.0, 12.0))


def test_scale_along_basis_scalar_s():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    z = np.zeros(2)
    xo, yo, zo = scale_along_basis(x, y, z, 2.0, basis.x, basis.y, basis.z)
    assert xo == pytest.approx(np.array([2.0, 0.0]))
    assert yo == pytest.approx(np.array([0.0, 2.0]))


def test_scale_along_basis_identity():
    x = np.array([1.0, 4.0])
    y = np.array([2.0, 5.0])
    z = np.array([3.0, 6.0])
    xo, yo, zo = scale_along_basis(x, y, z, 1.0, basis.x, basis.y, basis.z)
    assert xo == pytest.approx(x)
    assert yo == pytest.approx(y)
    assert zo == pytest.approx(z)
