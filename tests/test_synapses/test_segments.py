"""Geometry tests for point-to-segment projection."""

from __future__ import annotations

import numpy as np
import pytest

from neurosetta.utils.geometry_utils.segments import (
    project_points_to_segments,
    project_points_to_segments_bruteforce,
)


def test_point_on_segment():
    pts = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0]])
    edge, nearest, dist, t = project_points_to_segments(pts, starts, ends)
    assert edge[0] == 0
    assert t[0] == pytest.approx(0.5)
    assert dist[0] == pytest.approx(0.0)
    np.testing.assert_allclose(nearest[0], [0.5, 0.0, 0.0])


def test_point_beside_segment():
    pts = np.array([[0.5, 1.0, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0]])
    _, nearest, dist, t = project_points_to_segments(pts, starts, ends)
    assert t[0] == pytest.approx(0.5)
    assert dist[0] == pytest.approx(1.0)
    np.testing.assert_allclose(nearest[0], [0.5, 0.0, 0.0])


def test_nearest_at_source_endpoint():
    pts = np.array([[-1.0, 0.5, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0]])
    _, nearest, _, t = project_points_to_segments(pts, starts, ends)
    assert t[0] == pytest.approx(0.0)
    np.testing.assert_allclose(nearest[0], [0.0, 0.0, 0.0])


def test_nearest_at_target_endpoint():
    pts = np.array([[2.0, 0.5, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0]])
    _, nearest, _, t = project_points_to_segments(pts, starts, ends)
    assert t[0] == pytest.approx(1.0)
    np.testing.assert_allclose(nearest[0], [1.0, 0.0, 0.0])


def test_competing_edges_lowest_index_wins_on_tie():
    pts = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    ends = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    edge, _, dist, _ = project_points_to_segments(pts, starts, ends)
    assert edge[0] == 0
    assert dist[0] == pytest.approx(0.0)


def test_zero_length_edge():
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    starts = np.array([[0.0, 0.0, 0.0]])
    ends = np.array([[0.0, 0.0, 0.0]])
    _, nearest, dist, t = project_points_to_segments(pts, starts, ends)
    assert t[0] == pytest.approx(0.0)
    assert dist[0] == pytest.approx(1.0)
    np.testing.assert_allclose(nearest[0], [0.0, 0.0, 0.0])


def test_many_points_many_edges():
    rng = np.random.default_rng(0)
    starts = rng.normal(size=(50, 3))
    ends = starts + rng.normal(size=(50, 3)) * 0.2
    pts = rng.normal(size=(200, 3))
    edge, nearest, dist, t = project_points_to_segments_bruteforce(pts, starts, ends)
    assert edge.shape == (200,)
    assert nearest.shape == (200, 3)
    assert dist.shape == (200,)
    assert t.shape == (200,)
    assert np.all((t >= 0.0) & (t <= 1.0))
    assert np.all(dist >= 0.0)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_float_dtypes(dtype):
    pts = np.array([[0.25, 0.0, 0.0]], dtype=dtype)
    starts = np.array([[0.0, 0.0, 0.0]], dtype=dtype)
    ends = np.array([[1.0, 0.0, 0.0]], dtype=dtype)
    _, _, dist, t = project_points_to_segments(pts, starts, ends)
    assert t[0] == pytest.approx(0.25, abs=1e-5)
    assert dist[0] == pytest.approx(0.0, abs=1e-5)
