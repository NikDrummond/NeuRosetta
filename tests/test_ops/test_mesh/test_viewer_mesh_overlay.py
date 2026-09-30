"""Mesh overlay on Tree / Forest 3D plots and Viewer."""

from __future__ import annotations

import pytest
from vedo import Sphere

from neurosetta import Tree_mesh
from neurosetta.ops.plotting import Viewer
from neurosetta.ops.plotting.mesh_plot_utils import resolve_mesh_overlay
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


@pytest.fixture
def tree():
    return make_synthetic_tree(30, tree_id=7, units="micron")


@pytest.fixture
def neuron_mesh(tree):
    return Tree_mesh(ID=tree.ID, metadata={"units": "micron"}, mesh=Sphere(r=2.0))


def test_resolve_mesh_overlay_defaults_and_alias():
    assert resolve_mesh_overlay() is False
    assert resolve_mesh_overlay(mesh=True) is True
    assert resolve_mesh_overlay(show_mesh=True) is True
    assert resolve_mesh_overlay(mesh=False, show_mesh=False) is False
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_mesh_overlay(mesh=True, show_mesh=False)


def test_add_neuron_overlays_mesh(tree, neuron_mesh):
    tree.set_mesh(neuron_mesh)
    viewer = Viewer(offscreen=True)
    viewer.add_neuron(tree)
    n_skel = len(viewer._actors)
    viewer2 = Viewer(offscreen=True)
    viewer2.add_neuron(tree, show_mesh=True)
    assert len(viewer2._actors) > n_skel
    viewer.close()
    viewer2.close()


def test_add_neuron_warns_without_mesh(tree):
    viewer = Viewer(offscreen=True)
    with pytest.warns(UserWarning, match="no attached mesh"):
        viewer.add_neuron(tree, show_mesh=True)
    viewer.close()


def test_plot_3d_show_mesh_via_viewer(tree, neuron_mesh):
    tree.set_mesh(neuron_mesh)
    viewer = Viewer(offscreen=True)
    plot = tree.make_plot3d(cache=True)
    viewer.add(*plot.actors)
    actors = viewer.add_mesh(tree.mesh, alpha=0.35)
    assert len(actors) == 1
    viewer.close()
    assert resolve_mesh_overlay(show_mesh=True) is True


def test_forest_show_mesh_overlay():
    forest = make_synthetic_forest(2, n=20, units="micron", tree_ids=[1, 2])
    for t in forest:
        t.set_mesh(Tree_mesh(ID=t.ID, metadata={"units": "micron"}, mesh=Sphere(r=1.0)))
    v = Viewer(offscreen=True)
    v.add_forest(forest, show_mesh=True)
    # skeleton actors + at least one mesh actor per tree
    assert len(v._actors) >= 4
    v.close()


def test_forest_show_mesh_warns_when_empty():
    forest = make_synthetic_forest(2, n=15, units="micron")
    v = Viewer(offscreen=True)
    with pytest.warns(UserWarning, match="no trees"):
        v.add_forest(forest, show_mesh=True)
    v.close()
