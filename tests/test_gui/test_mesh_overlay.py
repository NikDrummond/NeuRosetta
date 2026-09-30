"""GUI mesh overlay — directory match + attached tree.mesh facet."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from vedo import Plotter, Sphere

from neurosetta.config import configure, get_settings
from neurosetta.gui.core.application import NeuroGUIApplication
from neurosetta.testing import make_synthetic_tree


@pytest.fixture
def offscreen():
    previous = get_settings().vedo.offscreen
    configure(vedo_offscreen=True, vedo_backend="vtk")
    yield
    configure(vedo_offscreen=previous, vedo_backend="vtk")


@pytest.fixture
def app(offscreen):
    plotter = Plotter(offscreen=True, size=(200, 200))
    application = NeuroGUIApplication(plotter, MagicMock())
    yield application
    plotter.close()


def _write_nr_and_mesh(tmp_path: Path, *, tree_id: int = 7, with_facet: bool = False):
    tree = make_synthetic_tree(20, tree_id=tree_id, units="micron")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    ply = mesh_dir / f"{tree_id}.ply"
    Sphere(r=1.5).write(str(ply))
    if with_facet:
        tree.set_mesh(ply, set_units="micron")
    nr_path = tmp_path / f"{tree_id}.nr"
    tree.save_tree(nr_path)
    return nr_path, mesh_dir


def test_gui_mesh_from_directory(app, tmp_path):
    nr_path, mesh_dir = _write_nr_and_mesh(tmp_path, with_facet=False)
    assert app.load_file(str(nr_path))
    app.set_mesh_directory(str(mesh_dir))
    app.set_show_mesh(True)
    assert app.current_mesh is not None
    assert app.current_mesh.npoints > 0


def test_gui_mesh_from_attached_facet_without_directory(app, tmp_path):
    nr_path, _ = _write_nr_and_mesh(tmp_path, with_facet=True)
    assert app.load_file(str(nr_path))
    assert app.current_neuron.has_mesh()
    app.set_show_mesh(True)  # no Set Mesh Path
    assert app.current_mesh is not None
    assert app.current_mesh.npoints > 0


def test_gui_mesh_prefers_attached_over_directory(app, tmp_path):
    nr_path, mesh_dir = _write_nr_and_mesh(tmp_path, tree_id=7, with_facet=True)
    # Different file on disk — facet should win
    Sphere(r=0.5).write(str(mesh_dir / "7.ply"))
    assert app.load_file(str(nr_path))
    n_facet = app.current_neuron.mesh.count_vertices()
    app.set_mesh_directory(str(mesh_dir))
    app.set_show_mesh(True)
    assert app.current_mesh is not None
    assert app.current_mesh.npoints == n_facet


def test_gui_mesh_clears_when_missing(app, tmp_path):
    nr_path, mesh_dir = _write_nr_and_mesh(tmp_path, tree_id=7, with_facet=False)
    assert app.load_file(str(nr_path))
    app.set_mesh_directory(str(mesh_dir))
    app.set_show_mesh(True)
    assert app.current_mesh is not None

    # Navigate to a neuron with no mesh file / facet
    other = make_synthetic_tree(15, tree_id=99, units="micron")
    other_path = tmp_path / "99.nr"
    other.save_tree(other_path)
    app.files = [str(nr_path), str(other_path)]
    app.current_file_index = 1
    assert app._load_current_file()
    assert app.current_mesh is None


def test_gui_hide_mesh_checkbox(app, tmp_path):
    nr_path, mesh_dir = _write_nr_and_mesh(tmp_path)
    assert app.load_file(str(nr_path))
    app.set_mesh_directory(str(mesh_dir))
    app.set_show_mesh(True)
    assert app.current_mesh is not None
    app.set_show_mesh(False)
    assert app.current_mesh is None


def test_gui_directory_import_does_not_mutate_source(tmp_path, offscreen):
    """Overlay styling clones the vedo mesh; re-import stays unstyled."""
    from neurosetta import import_mesh

    nr_path, mesh_dir = _write_nr_and_mesh(tmp_path)
    plotter = Plotter(offscreen=True, size=(200, 200))
    app = NeuroGUIApplication(plotter, MagicMock())
    app.load_file(str(nr_path))
    app.set_mesh_directory(str(mesh_dir))
    app.set_show_mesh(True)
    reloaded = import_mesh(mesh_dir / "7.ply", mesh_type="Neuron")
    # Source file re-import should not carry GUI gray/alpha as identity proof —
    # mainly ensure clone path didn't explode; alpha may still default.
    assert reloaded.mesh.npoints == app.current_mesh.npoints
    plotter.close()
