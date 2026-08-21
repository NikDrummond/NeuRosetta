"""Tests for geometry_utils.pca."""

import numpy as np
import pytest

from neurosetta.utils.geometry_utils import pca


def test_covariance_xyz_line_along_x():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.zeros(4)
    z = np.zeros(4)
    cov = pca.covariance_xyz(x, y, z)
    assert cov[0, 0] == pytest.approx(1.666667, rel=1e-4)
    assert cov[1, 1] == pytest.approx(0.0, abs=1e-12)
    assert cov[2, 2] == pytest.approx(0.0, abs=1e-12)


def test_covariance_eigh_orders_descending():
    cov = np.diag([1.0, 3.0, 2.0])
    evals, evecs = pca.covariance_eigh(cov)
    assert evals[0] >= evals[1] >= evals[2]
    assert evals[0] == pytest.approx(3.0)


def test_eig_decomp_major_axis_along_x():
    x = np.linspace(0.0, 10.0, 50)
    y = np.zeros(50)
    z = np.zeros(50)
    evals, evecs = pca.eig_decomp(x, y, z)
    major = evecs[:, 0]
    assert abs(major[0]) == pytest.approx(1.0, abs=1e-6)


def test_eig_decomp_normalized_evals():
    x = np.linspace(0.0, 10.0, 20)
    y = np.linspace(0.0, 1.0, 20)
    z = np.zeros(20)
    evals, _ = pca.eig_decomp(x, y, z, norm=True)
    assert evals.sum() == pytest.approx(1.0)


def test_robust_covariance_runs():
    x = np.array([0.0, 1.0, 2.0, 100.0])
    y = np.zeros(4)
    z = np.zeros(4)
    cov = pca.robust_covariance_xyz(x, y, z)
    assert cov.shape == (3, 3)


def test_robust_covariance_downweights_outlier():
    x = np.linspace(0.0, 1.0, 20)
    y = np.zeros(20)
    z = np.zeros(20)
    y[-1] = 100.0
    sample = pca.covariance_xyz(x, y, z)
    robust = pca.robust_covariance_xyz(x, y, z)
    assert robust[1, 1] < sample[1, 1]


def test_eig_decomp_robust():
    x = np.linspace(0.0, 10.0, 30)
    y = np.zeros(30)
    z = np.zeros(30)
    evals, evecs = pca.eig_decomp(x, y, z, robust=True)
    assert evals.shape == (3,)
    assert abs(evecs[0, 0]) == pytest.approx(1.0, abs=1e-5)


def test_covariance_eigh_right_handed():
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 1.0, 0.0])
    v2 = np.array([0.0, 0.0, -1.0])
    vecs = np.column_stack([v0, v1, v2])
    evals = np.array([5.0, 3.0, 1.0])
    cov = vecs @ np.diag(evals) @ vecs.T
    _, evecs = pca.covariance_eigh(cov)
    assert np.linalg.det(evecs) > 0.0


def test_eig_decomp_rejects_2d():
    with pytest.raises(ValueError, match="shape"):
        pca.eig_decomp(np.zeros((2, 3)), np.zeros(3), np.zeros(3))
