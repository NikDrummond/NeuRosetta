"""Parallel pairwise topological distance matrix tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurosetta.testing import make_synthetic_forest
from neurosetta.utils.topology import pairwise_distances


@pytest.fixture
def forest_ready():
    forest = make_synthetic_forest(n_trees=4, n=35, seed=21)
    for t in forest:
        t.get_root_distance(bind=True)
    return forest


def test_pairwise_barcode_serial_vs_parallel_equal():
    diags = [
        np.array([[0.0, 1.0], [0.5, 2.0]]),
        np.array([[0.0, 1.5]]),
        np.array([[0.2, 0.8], [1.0, 3.0]]),
        np.array([[0.0, 2.5]]),
    ]
    a = pairwise_distances(diags, metric="barcode", n_jobs=1)
    b = pairwise_distances(diags, metric="barcode", n_jobs=2)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert a.shape == (4, 4)
    np.testing.assert_allclose(np.diag(a), 0.0)
    np.testing.assert_allclose(a, a.T)


def test_forest_topology_matrix_parallel_false_vs_true(forest_ready):
    forest = forest_ready
    serial = forest.topology_distance_matrix(
        func="Root_distance",
        metric="barcode",
        parallel=False,
        show_progress=False,
    )
    parallel = forest.topology_distance_matrix(
        func="Root_distance",
        metric="barcode",
        parallel=True,
        max_workers=2,
        show_progress=False,
    )
    np.testing.assert_allclose(serial, parallel, atol=1e-12)
    assert serial.shape == (len(forest), len(forest))


@pytest.mark.skipif(
    importlib.util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)
def test_forest_bottleneck_serial_vs_parallel(forest_ready):
    forest = forest_ready
    serial = forest.topology_distance_matrix(
        func="Root_distance",
        metric="bottleneck",
        parallel=False,
        show_progress=False,
    )
    parallel = forest.topology_distance_matrix(
        func="Root_distance",
        metric="bottleneck",
        parallel=True,
        max_workers=2,
        show_progress=False,
    )
    np.testing.assert_allclose(serial, parallel, atol=1e-10)


def test_persistence_diagrams_accepts_parallel_kwargs(forest_ready):
    diags = forest_ready.persistence_diagrams(
        func="Root_distance",
        parallel=True,
        max_workers=2,
        show_progress=False,
    )
    assert len(diags) == len(forest_ready)
    assert all(d.ndim == 2 and d.shape[1] == 2 for d in diags)
