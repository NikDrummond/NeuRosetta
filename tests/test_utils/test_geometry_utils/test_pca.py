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
