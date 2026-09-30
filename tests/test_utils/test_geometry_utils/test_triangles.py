"""Tests for point-to-triangle projection."""

from __future__ import annotations

import numpy as np
import pytest

from neurosetta.utils.geometry_utils.triangles import (
    project_points_to_triangles,
    project_points_to_triangles_bruteforce,
)


def test_point_above_triangle_centroid():
    # Equilateral-ish right triangle in xy-plane
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 0.0]])
    c = np.array([[0.0, 2.0, 0.0]])
    # Centroid is (2/3, 2/3, 0); point 1 unit above
    pts = np.array([[2.0 / 3.0, 2.0 / 3.0, 1.0]])
    face, closest, dist = project_points_to_triangles_bruteforce(pts, a, b, c)
    assert face[0] == 0
    assert dist[0] == pytest.approx(1.0)
    assert closest[0] == pytest.approx([2.0 / 3.0, 2.0 / 3.0, 0.0])


def test_point_at_vertex():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    c = np.array([[0.0, 1.0, 0.0]])
    pts = np.array([[0.0, 0.0, 0.0]])
    _, closest, dist = project_points_to_triangles(pts, a, b, c)
    assert dist[0] == pytest.approx(0.0)
    assert closest[0] == pytest.approx([0.0, 0.0, 0.0])


def test_point_outside_projects_to_edge():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[2.0, 0.0, 0.0]])
    c = np.array([[0.0, 2.0, 0.0]])
    # Outside near midpoint of AB
    pts = np.array([[1.0, -1.0, 0.0]])
    _, closest, dist = project_points_to_triangles(pts, a, b, c, method="bruteforce")
    assert dist[0] == pytest.approx(1.0)
    assert closest[0] == pytest.approx([1.0, 0.0, 0.0])
