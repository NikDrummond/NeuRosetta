"""Tests for barcode density distance and GUDHI distance wrappers."""

from __future__ import annotations

import numpy as np
import pytest

from neurosetta.utils.topology import (
    barcode_distance,
    bottleneck_distance,
    pairwise_distances,
    wasserstein_distance,
)


def _brute_force_barcode_distance(a: np.ndarray, b: np.ndarray, n_grid: int = 5001) -> float:
    """Slow reference: integrate |dens_A - dens_B| on a dense grid."""
    if a.size == 0 and b.size == 0:
        return 0.0
    parts = []
    if a.size:
        parts.append(a.ravel())
    if b.size:
        parts.append(b.ravel())
    xs = np.concatenate(parts)
    lo, hi = float(xs.min()), float(xs.max())
    if hi <= lo:
        return 0.0
    grid = np.linspace(lo, hi, n_grid)
    dx = grid[1] - grid[0]

    def density(diag, x):
        if diag.size == 0:
            return np.zeros_like(x)
        left = np.minimum(diag[:, 0], diag[:, 1])[:, None]
        right = np.maximum(diag[:, 0], diag[:, 1])[:, None]
        # half-open [lo, hi): match event-sweep convention
        return ((x[None, :] >= left) & (x[None, :] < right)).sum(axis=0)

    diff = np.abs(density(a, grid) - density(b, grid))
    # exclude the last sample (zero-width)
    return float(diff[:-1].sum() * dx)


def test_barcode_identical_zero():
    d = np.array([[0.0, 1.0], [0.5, 2.0]])
    assert barcode_distance(d, d) == 0.0


def test_barcode_empty_vs_empty():
    e = np.zeros((0, 2))
    assert barcode_distance(e, e) == 0.0


def test_barcode_symmetry():
    a = np.array([[0.0, 2.0]])
    b = np.array([[1.0, 3.0]])
    assert barcode_distance(a, b) == pytest.approx(barcode_distance(b, a))


def test_barcode_analytic_simple():
    # A: [0, 2] density 1 on [0,2); B empty → integral = 2
    a = np.array([[0.0, 2.0]])
    b = np.zeros((0, 2))
    assert barcode_distance(a, b) == pytest.approx(2.0)
    # A: [0,2], B: [1,3] → dens_diff: [0,1)=1, [1,2)=0, [2,3)=1 → area 2
    b2 = np.array([[1.0, 3.0]])
    assert barcode_distance(a, b2) == pytest.approx(2.0)


def test_barcode_zero_length_ignored():
    a = np.array([[1.0, 1.0], [0.0, 2.0]])
    b = np.array([[0.0, 2.0]])
    assert barcode_distance(a, b) == pytest.approx(0.0)


def test_barcode_matches_brute_force_random():
    rng = np.random.default_rng(0)
    for _ in range(30):
        n, m = int(rng.integers(0, 6)), int(rng.integers(0, 6))
        a = rng.random((n, 2))
        b = rng.random((m, 2))
        # sort endpoints
        a = np.sort(a, axis=1)
        b = np.sort(b, axis=1)
        got = barcode_distance(a, b)
        ref = _brute_force_barcode_distance(a, b)
        assert got == pytest.approx(ref, abs=0.05)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)
def test_bottleneck_matches_gudhi():
    import gudhi

    a = np.array([[0.0, 1.0], [1.0, 2.0]])
    b = np.array([[0.0, 1.5]])
    assert bottleneck_distance(a, b) == pytest.approx(gudhi.bottleneck_distance(a, b))


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)
def test_wasserstein_matches_gudhi_hera():
    from gudhi.hera import wasserstein_distance as gudhi_w

    a = np.array([[0.0, 1.0], [1.0, 2.0]])
    b = np.array([[0.0, 1.5]])
    assert wasserstein_distance(a, b) == pytest.approx(gudhi_w(a, b, order=1.0, internal_p=np.inf))


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)
def test_pairwise_square_symmetric():
    diags = [
        np.array([[0.0, 1.0]]),
        np.array([[0.0, 2.0]]),
        np.array([[1.0, 1.5]]),
    ]
    for metric in ("bottleneck", "wasserstein", "barcode"):
        mat = pairwise_distances(diags, metric=metric)
        assert mat.shape == (3, 3)
        np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-12)
        np.testing.assert_allclose(mat, mat.T, atol=1e-12)
