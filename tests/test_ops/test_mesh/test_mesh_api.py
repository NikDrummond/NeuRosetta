"""Tests for mesh geometry ops and plot helpers."""

from __future__ import annotations

import numpy as np
import pytest
from vedo import Sphere

from neurosetta import Forest_mesh, Neuropil, Tree_mesh, configure, get_settings
from neurosetta.ops.mesh import (
    count_faces,
    count_vertices,
    distance_to_surface,
    get_bounds,
    get_vertex_coordinates,
)
from neurosetta.ops.plotting import Viewer, make_mesh_actor, plot_mesh
from neurosetta.ops.plotting.plot_mesh import plot_mesh as plot_mesh_fn
from neurosetta.utils.vedo_utils.surface_distances import surface_distance


@pytest.fixture
def neuron_mesh() -> Tree_mesh:
    return Tree_mesh(ID=1, metadata={"units": "micron"}, mesh=Sphere(r=2.0))


@pytest.fixture
def neuropil() -> Neuropil:
    return Neuropil(ID="AL", metadata={"units": "micron"}, mesh=Sphere(r=3.0))


@pytest.fixture
def _offscreen():
    previous = get_settings().vedo.offscreen
    configure(vedo_offscreen=True)
    yield
    configure(vedo_offscreen=previous)


def test_geometry_queries(neuron_mesh):
    assert count_vertices(neuron_mesh) == neuron_mesh.mesh.npoints
    assert count_faces(neuron_mesh) > 0
    coords = get_vertex_coordinates(neuron_mesh)
    assert coords.shape == (neuron_mesh.count_vertices(), 3)
    bounds = get_bounds(neuron_mesh)
    assert bounds.shape == (6,)
    assert bounds[0] < bounds[1]


def test_distance_to_surface_matches_utils(neuron_mesh):
    pts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    d_api = distance_to_surface(neuron_mesh, pts)
    d_raw, _ = surface_distance(pts, neuron_mesh.mesh)
    assert np.allclose(d_api, d_raw)
    assert np.allclose(neuron_mesh.distance_to_surface(pts), d_api)


def test_make_mesh_actor_does_not_mutate(neuron_mesh):
    before = np.asarray(neuron_mesh.mesh.vertices).copy()
    actor = make_mesh_actor(neuron_mesh, c="red", alpha=0.2)
    assert actor is not neuron_mesh.mesh
    assert np.allclose(neuron_mesh.mesh.vertices, before)
    assert actor.npoints == neuron_mesh.count_vertices()


def test_plot_mesh_return_viewer(neuron_mesh, _offscreen):
    viewer = plot_mesh(neuron_mesh, return_viewer=True, alpha=0.3, c="cyan")
    assert isinstance(viewer, Viewer)
    assert len(viewer._actors) == 1
    viewer.close()


def test_viewer_add_mesh_collection(neuron_mesh, neuropil, _offscreen):
    forest = Forest_mesh(
        [
            neuron_mesh,
            Tree_mesh(ID=2, metadata={"units": "micron"}, mesh=Sphere(r=1.5)),
        ]
    )
    viewer = Viewer()
    actors = viewer.add_mesh(forest, alpha=0.4)
    assert len(actors) == 2
    viewer.close()


def test_show_3d_bound_to_plot_mesh(neuron_mesh, neuropil):
    assert type(neuron_mesh).show_3d is plot_mesh_fn
    assert type(neuropil).show_3d is plot_mesh_fn
    assert Forest_mesh.show_3d is plot_mesh_fn


def test_export_mesh_bound(neuron_mesh, tmp_path):
    out = neuron_mesh.export_mesh(tmp_path / "n.ply")
    assert out.exists()
