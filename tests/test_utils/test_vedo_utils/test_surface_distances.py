"""Tests for vedo_utils.surface_distances."""

import numpy as np
import pytest
from vedo import Mesh

from NeuRosetta.utils.vedo_utils.surface_distances import (
    _compute_face_centroids_and_normals,
    _get_face_subset_from_dot,
    build_submesh,
    mesh_surface_depth,
    surface_distance,
)


@pytest.fixture
def unit_triangle_mesh():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    return Mesh([verts, faces])


def test_build_submesh_none_returns_original(unit_triangle_mesh):
    assert build_submesh(unit_triangle_mesh, face_indices=None) is unit_triangle_mesh


def test_build_submesh_single_face(unit_triangle_mesh):
    sub = build_submesh(unit_triangle_mesh, face_indices=[0])
    assert len(sub.vertices) == 3
    assert len(sub.cells) == 1


def test_surface_distance_to_vertex(unit_triangle_mesh):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dists, closest = surface_distance(points, unit_triangle_mesh)
    assert dists.shape == (2,)
    assert closest.shape == (2, 3)
    assert dists[0] == pytest.approx(0.0)


def test_compute_face_centroids_and_normals(unit_triangle_mesh):
    centroids, normals = _compute_face_centroids_and_normals(unit_triangle_mesh)
    assert centroids.shape == (1, 3)
    assert normals.shape == (1, 3)
    assert np.linalg.norm(normals[0]) == pytest.approx(1.0)


def test_get_face_subset_from_dot_both(unit_triangle_mesh):
    inner, outer = _get_face_subset_from_dot(unit_triangle_mesh, t=0.0, face="both")
    assert len(inner[0]) + len(outer[0]) <= 1


def test_get_face_subset_invalid_face_raises(unit_triangle_mesh):
    with pytest.raises(AttributeError, match="face must be"):
        _get_face_subset_from_dot(unit_triangle_mesh, t=0.0, face="sideways")


def test_mesh_surface_depth_runs():
    # Two coplanar triangles with opposite winding → inner and outer subsets
    verts = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=int)
    mesh = Mesh([verts, faces])
    points = np.array([[0.25, 0.25, 0.5], [0.5, 0.1, 0.5]])
    depth = mesh_surface_depth(mesh, points, t=0.0, surface="outer", norm=True)
    assert depth.shape == (2,)
    assert np.all(np.isfinite(depth))
