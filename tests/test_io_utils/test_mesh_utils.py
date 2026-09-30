"""Mesh I/O and Forest_mesh / Neuropils contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from vedo import Sphere

from neurosetta import (
    Forest_mesh,
    Neuropil,
    Neuropils,
    Tree_mesh,
    export_mesh,
    import_mesh,
)
from neurosetta.utils.units import is_dimensionless


def _write_sphere(path: Path, *, r: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sphere(r=r).write(str(path))
    return path


@pytest.fixture
def mesh_file(tmp_path: Path) -> Path:
    return _write_sphere(tmp_path / "neuron42.ply")


@pytest.fixture
def mesh_dir(tmp_path: Path) -> Path:
    d = tmp_path / "meshes"
    _write_sphere(d / "a.ply", r=1.0)
    _write_sphere(d / "b.ply", r=1.5)
    return d


def test_import_single_neuron_metadata(mesh_file: Path):
    m = import_mesh(mesh_file, mesh_type="Neuron")
    assert isinstance(m, Tree_mesh)
    assert m.ID == "neuron42"
    assert m.metadata["file_path"] == str(mesh_file)
    assert "units" in m.metadata
    assert is_dimensionless(m.metadata["units"])
    assert m.list_properties() == []
    assert "file_path" in m.list_meta()
    assert "file_path" in m.list_meta(include_protected=True)


def test_import_single_neuropil(mesh_file: Path):
    m = import_mesh(mesh_file, mesh_type="Neuropil")
    assert isinstance(m, Neuropil)
    assert m.ID == "neuron42"
    assert m.metadata["file_path"] == str(mesh_file)


def test_import_set_units(mesh_file: Path):
    m = import_mesh(mesh_file, set_units="nm")
    assert m.metadata["units"] == "nanometer" or "nano" in str(m.metadata["units"])
    # Canonical pint form
    from neurosetta.utils.units import normalize_units_str

    assert normalize_units_str(m.metadata["units"]) == normalize_units_str("nm")


def test_import_bad_mesh_type(mesh_file: Path):
    with pytest.raises(ValueError, match="mesh_type"):
        import_mesh(mesh_file, mesh_type="Banana")  # type: ignore[arg-type]


def test_import_directory_forest_metadata(mesh_dir: Path):
    forest = import_mesh(mesh_dir, mesh_type="Neuron", set_units="um")
    assert isinstance(forest, Forest_mesh)
    assert len(forest) == 2
    assert set(forest.ids()) == {"a", "b"}
    assert "Forest_mesh" in repr(forest)
    for m in forest:
        assert m.metadata.get("file_path")
        assert not is_dimensionless(m.metadata["units"])
    # Forest metadata helpers must not crash on meshes.
    keys = forest.list_meta()
    assert "units" in keys
    assert "file_path" in keys
    assert forest.list_properties() == [[], []]
    summary = forest.meta_summary()
    assert summary["units"] == 2


def test_import_directory_neuropils(mesh_dir: Path):
    col = import_mesh(mesh_dir, mesh_type="Neuropil")
    assert isinstance(col, Neuropils)
    assert "Neuropils" in repr(col)
    assert len(col) == 2


def test_forest_mesh_rejects_wrong_member_type(mesh_file: Path):
    neuron = import_mesh(mesh_file, mesh_type="Neuron")
    neuropil = import_mesh(mesh_file, mesh_type="Neuropil")
    with pytest.raises(TypeError, match="Tree_mesh"):
        Forest_mesh([neuropil])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="Neuropil"):
        Neuropils([neuron])  # type: ignore[list-item]


def test_forest_mesh_build_3d_raises(mesh_dir: Path):
    forest = import_mesh(mesh_dir, mesh_type="Neuron")
    with pytest.raises(NotImplementedError, match="build_3d"):
        forest.build_3d()


def test_export_import_roundtrip(mesh_file: Path, tmp_path: Path):
    original = import_mesh(mesh_file, mesh_type="Neuron", set_units="nm")
    out = tmp_path / "roundtrip" / "out.ply"
    written = export_mesh(original, out)
    assert Path(written).exists()
    loaded = import_mesh(written, mesh_type="Neuron", set_units="nm")
    assert loaded.ID == "out"
    assert loaded.mesh.npoints == original.mesh.npoints


def test_export_forest(mesh_dir: Path, tmp_path: Path):
    forest = import_mesh(mesh_dir, mesh_type="Neuron")
    out_dir = tmp_path / "exported"
    paths = export_mesh(forest, out_dir)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    reloaded = import_mesh(out_dir, mesh_type="Neuron")
    assert len(reloaded) == 2


def test_export_forest_to_file_raises(mesh_dir: Path, tmp_path: Path):
    forest = import_mesh(mesh_dir, mesh_type="Neuron")
    with pytest.raises(ValueError, match="single file"):
        export_mesh(forest, tmp_path / "nope.ply")
