"""Tests for geometry_utils.basis."""

import numpy as np
import pytest

from NeuRosetta.utils.geometry_utils.basis import basis


def test_basis_vectors():
    assert np.allclose(basis.x, [1.0, 0.0, 0.0])
    assert np.allclose(basis.y, [0.0, 1.0, 0.0])
    assert np.allclose(basis.z, [0.0, 0.0, 1.0])
    assert np.allclose(basis.neg_x, [-1.0, 0.0, 0.0])


def test_basis_vectors_readonly():
    with pytest.raises(ValueError):
        basis.x[0] = 5.0
