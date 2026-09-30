"""Tests for the tree.mesh neuron-surface facet."""

from __future__ import annotations

import numpy as np
import pytest
from vedo import Sphere

from neurosetta import Neuropil, Tree_mesh
from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.units import normalize_units_str


@pytest.fixture
def tree():
    return make_synthetic_tree(40, tree_id=7, units="micron")


@pytest.fixture
def neuron_mesh(tree):
    return Tree_mesh(ID=tree.ID, metadata={"units": "micron"}, mesh=Sphere(r=2.0))


def test_set_get_clear_mesh(tree, neuron_mesh):
    assert not tree.has_mesh()
    assert tree.mesh is None
    bound = tree.set_mesh(neuron_mesh)
    assert tree.has_mesh()
    assert tree.mesh is bound
    assert bound.ID == tree.ID
    assert normalize_units_str(bound.get_units()) == normalize_units_str("micron")
    tree.clear_mesh()
    assert tree.mesh is None


def test_mesh_property_setter(tree, neuron_mesh):
    tree.mesh = neuron_mesh
    assert tree.has_mesh()
    tree.mesh = None
    assert not tree.has_mesh()


def test_id_mismatch_raises(tree):
    bad = Tree_mesh(ID=999, metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    with pytest.raises(ValueError, match="does not match"):
        tree.set_mesh(bad)


def test_neuropil_rejected(tree):
    npil = Neuropil(ID="AL", metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    with pytest.raises(TypeError, match="neuron mesh"):
        tree.set_mesh(npil)


def test_units_mismatch_raises(tree):
    mesh = Tree_mesh(ID=tree.ID, metadata={"units": "nm"}, mesh=Sphere(r=1.0))
    with pytest.raises(ValueError, match="incompatible"):
        tree.set_mesh(mesh)


def test_vedo_mesh_coercion(tree):
    tree.set_mesh(Sphere(r=1.5))
    assert tree.mesh is not None
    assert tree.mesh.ID == tree.ID
    assert tree.mesh.count_vertices() > 0


def test_copy_isolates_mesh(tree, neuron_mesh):
    tree.set_mesh(neuron_mesh)
    clone = tree.copy()
    assert clone.has_mesh()
    assert clone.mesh is not tree.mesh
    before = np.asarray(tree.mesh.mesh.vertices).copy()
    clone.mesh.mesh.vertices = before * 2
    assert np.allclose(tree.mesh.mesh.vertices, before)


def test_nr_roundtrip(tree, neuron_mesh, tmp_path):
    tree.set_mesh(neuron_mesh)
    n_before = tree.mesh.count_vertices()
    path = tmp_path / "7.nr"
    tree.save_tree(path)
    from neurosetta.io import load

    loaded = load(path)
    assert loaded.has_mesh()
    assert loaded.mesh.ID == loaded.ID
    assert loaded.mesh.count_vertices() == n_before
    assert normalize_units_str(loaded.mesh.get_units()) == normalize_units_str("micron")


def test_convert_units_scales_mesh(tree, neuron_mesh):
    tree.set_mesh(neuron_mesh)
    before = np.asarray(tree.mesh.mesh.vertices).copy()
    tree.convert_units("nm")
    assert normalize_units_str(tree.mesh.get_units()) == normalize_units_str("nm")
    assert np.allclose(tree.mesh.mesh.vertices, before * 1000.0)


def test_translate_moves_mesh(tree, neuron_mesh):
    tree.set_mesh(neuron_mesh)
    before = np.asarray(tree.mesh.mesh.vertices).copy()
    tree.translate_coordinates(10.0, 0.0, 0.0)
    after = np.asarray(tree.mesh.mesh.vertices)
    assert np.allclose(after[:, 0], before[:, 0] + 10.0)
    assert np.allclose(after[:, 1:], before[:, 1:])


def test_set_mesh_from_path(tree, tmp_path):
    from vedo import Sphere

    path = tmp_path / f"{tree.ID}.ply"
    Sphere(r=1.0).write(str(path))
    bound = tree.set_mesh(path, set_units="micron")
    assert tree.has_mesh()
    assert bound.ID == tree.ID
    assert normalize_units_str(bound.get_units()) == normalize_units_str("micron")


def test_set_mesh_path_rejects_directory(tree, tmp_path):
    d = tmp_path / "meshes"
    d.mkdir()
    with pytest.raises(TypeError, match="directory"):
        tree.set_mesh(d)


def test_set_mesh_units_kwargs_require_path(tree, neuron_mesh):
    with pytest.raises(ValueError, match="only apply when data is a path"):
        tree.set_mesh(neuron_mesh, set_units="micron")
