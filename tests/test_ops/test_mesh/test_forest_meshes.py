"""Forest.set_meshes — ID-matched batch attach."""

from __future__ import annotations

from pathlib import Path

import pytest
from vedo import Sphere

from neurosetta import Forest_mesh, Tree_mesh, export_mesh, set_meshes
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


def _write_sphere(path: Path, *, r: float = 1.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Sphere(r=r).write(str(path))
    return path


@pytest.fixture
def forest():
    return make_synthetic_forest(2, n=20, units="micron", tree_ids=[1, 2])


@pytest.fixture
def mesh_dir(tmp_path: Path) -> Path:
    d = tmp_path / "meshes"
    # stems match forest tree IDs (str coerce: "1" ↔ 1)
    _write_sphere(d / "1.ply", r=1.0)
    _write_sphere(d / "2.ply", r=1.5)
    return d


def test_set_meshes_from_directory(forest, mesh_dir):
    attached = forest.set_meshes(mesh_dir, set_units="micron", missing="error")
    assert set(attached) == {1, 2}
    for t in forest:
        assert t.has_mesh()
        assert t.mesh.ID in (t.ID, str(t.ID)) or str(t.mesh.ID) == str(t.ID)


def test_set_meshes_package_export(forest, mesh_dir):
    attached = set_meshes(forest, mesh_dir, set_units="micron")
    assert len(attached) == 2


def test_set_meshes_int_str_id_match():
    tree = make_synthetic_tree(20, tree_id=7, units="micron")
    from neurosetta import Forest

    forest = Forest([tree])
    mesh = Tree_mesh(ID="7", metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    attached = forest.set_meshes([mesh], missing="error")
    assert 7 in attached
    assert forest[0].has_mesh()


def test_set_meshes_mapping_key_as_id(forest):
    # Mapping key overrides mesh.ID for matching
    m = Tree_mesh(ID="wrong", metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    attached = forest.set_meshes({1: m}, missing="ignore", unused="ignore")
    assert 1 in attached
    assert forest.by_id(1).has_mesh()


def test_set_meshes_forest_mesh_iterable(forest):
    meshes = Forest_mesh(
        [
            Tree_mesh(ID=1, metadata={"units": "micron"}, mesh=Sphere(r=1.0)),
            Tree_mesh(ID=2, metadata={"units": "micron"}, mesh=Sphere(r=1.0)),
        ]
    )
    attached = forest.set_meshes(meshes, missing="error")
    assert len(attached) == 2


def test_set_meshes_missing_warn_and_error(forest, tmp_path):
    d = tmp_path / "one"
    _write_sphere(d / "1.ply")
    with pytest.warns(UserWarning, match="No mesh found for tree ID 2"):
        forest.set_meshes(d, set_units="micron", missing="warn")
    with pytest.raises(ValueError, match="No mesh found for tree ID 2"):
        forest.set_meshes(d, set_units="micron", missing="error")


def test_set_meshes_unused_warn_and_error(forest, mesh_dir):
    _write_sphere(mesh_dir / "99.ply")
    with pytest.warns(UserWarning, match="unused"):
        forest.set_meshes(mesh_dir, set_units="micron", unused="warn")
    with pytest.raises(ValueError, match="unused"):
        forest.set_meshes(mesh_dir, set_units="micron", unused="error")


def test_set_meshes_ambiguous_ids(forest):
    meshes = [
        Tree_mesh(ID=1, metadata={"units": "micron"}, mesh=Sphere(r=1.0)),
        Tree_mesh(ID="1", metadata={"units": "micron"}, mesh=Sphere(r=1.0)),
    ]
    with pytest.raises(ValueError, match="Multiple meshes"):
        forest.set_meshes(meshes, missing="ignore")


def test_set_meshes_path_kwargs_only_for_path(forest):
    mesh = Tree_mesh(ID=1, metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    with pytest.raises(ValueError, match="only apply when meshes is a path"):
        forest.set_meshes([mesh], set_units="micron")


def test_set_meshes_export_roundtrip(forest, tmp_path):
    for t in forest:
        t.set_mesh(Tree_mesh(ID=t.ID, metadata={"units": "micron"}, mesh=Sphere(r=1.0)))
    out = tmp_path / "exported"
    out.mkdir()
    for t in forest:
        export_mesh(t.mesh, out / f"{t.ID}.ply")
    for t in forest:
        t.clear_mesh()
    attached = forest.set_meshes(out, set_units="micron", missing="error")
    assert len(attached) == 2
