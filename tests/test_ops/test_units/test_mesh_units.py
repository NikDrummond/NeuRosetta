"""Tests for mesh unit tracking and conversion."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from vedo import Sphere

from neurosetta import Forest_mesh, Neuropil, Neuropils, Tree_mesh, import_mesh
from neurosetta.ops.units import mesh_units as mu
from neurosetta.utils.units import DEFAULT_UNITS, normalize_units_str


def _sphere_mesh(ID: str = "m", *, units: str | None = None) -> Tree_mesh:
    meta: dict = {"units": units} if units is not None else {}
    return Tree_mesh(ID=ID, metadata=meta, mesh=Sphere(r=1.0))


def test_get_units_defaults_to_dimensionless():
    m = _sphere_mesh()
    assert mu.get_units(m) == DEFAULT_UNITS
    assert m.get_units() == DEFAULT_UNITS


def test_set_units_normalizes_without_scaling():
    m = _sphere_mesh(units="nm")
    before = np.asarray(m.mesh.vertices).copy()
    m.set_units("um", convert=False)
    assert m.get_units() == normalize_units_str("um")
    assert np.allclose(m.mesh.vertices, before)


def test_set_units_convert_scales_vertices():
    m = _sphere_mesh(units="nm")
    before = np.asarray(m.mesh.vertices).copy()
    m.set_units("um", convert=True)
    assert m.get_units() == normalize_units_str("um")
    assert np.allclose(m.mesh.vertices, before * 0.001)


def test_convert_units_scales_geometry():
    m = _sphere_mesh(units="nm")
    before = np.asarray(m.mesh.vertices).copy()
    m.convert_units("micron")
    assert m.get_units() == "micron"
    assert np.allclose(m.mesh.vertices, before * 0.001)


def test_convert_units_in_place_false_returns_copy():
    m = _sphere_mesh(units="nm")
    original = m.get_units()
    before = np.asarray(m.mesh.vertices).copy()
    out = m.convert_units("micron", in_place=False)
    assert out is not m
    assert m.get_units() == original
    assert np.allclose(m.mesh.vertices, before)
    assert out.get_units() == "micron"
    assert np.allclose(out.mesh.vertices, before * 0.001)


def test_convert_dimensionless_raises():
    m = _sphere_mesh()
    with pytest.raises(ValueError, match="dimensionless"):
        m.convert_units("nm")


def test_check_units_defined_raises():
    m = _sphere_mesh()
    with pytest.raises(ValueError, match="dimensionless"):
        m.check_units_defined()


def test_voxel_roundtrip_and_snap():
    m = _sphere_mesh(units="nm")
    # Place vertices off-integer so snap is observable after convert.
    verts = np.asarray(m.mesh.vertices, dtype=np.float64)
    m.mesh.vertices = verts + 0.4

    m.convert_units("voxel", voxel_size=10.0, voxel_unit="nm")
    assert m.get_units() == "voxel"
    assert m.get_voxel_spec() == (10.0, normalize_units_str("nm"))

    continuous = np.asarray(m.mesh.vertices).copy()
    m.snap_voxel_coordinates(method="floor")
    assert np.allclose(m.mesh.vertices, np.floor(continuous))


def test_neuropil_units_api():
    n = Neuropil(ID="AL", metadata={"units": "nm"}, mesh=Sphere(r=2.0))
    before = np.asarray(n.mesh.vertices).copy()
    n.convert_units("um")
    assert n.get_units() == normalize_units_str("um")
    assert np.allclose(n.mesh.vertices, before * 0.001)


def test_harmonize_forest_mesh_warns_and_converts():
    a = _sphere_mesh("a", units="nm")
    b = Tree_mesh(ID="b", metadata={"units": "micron"}, mesh=Sphere(r=1.5))
    forest = Forest_mesh([a, b])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        forest.harmonize_forest_units()

    assert any("mixed units" in str(w.message) for w in caught)
    assert a.get_units() == "micron"
    assert b.get_units() == "micron"


def test_ensure_forest_mesh_raises_on_dimensionless():
    a = _sphere_mesh("a", units="nm")
    b = _sphere_mesh("b")
    forest = Forest_mesh([a, b])
    with (
        pytest.warns(UserWarning, match="dimensionless"),
        pytest.raises(ValueError, match="dimensionless"),
    ):
        forest.ensure_forest_units()


def test_neuropils_harmonize():
    a = Neuropil(ID="a", metadata={"units": "nm"}, mesh=Sphere(r=1.0))
    b = Neuropil(ID="b", metadata={"units": "micron"}, mesh=Sphere(r=1.0))
    col = Neuropils([a, b])
    with pytest.warns(UserWarning, match="mixed units"):
        col.harmonize_forest_units()
    assert a.get_units() == "micron"
    assert b.get_units() == "micron"


def test_import_set_units_uses_mesh_units(tmp_path: Path):
    path = tmp_path / "n.ply"
    Sphere(r=1.0).write(str(path))
    m = import_mesh(path, set_units="nm")
    assert m.get_units() == normalize_units_str("nm")
    with pytest.raises(ValueError, match="voxel"):
        import_mesh(path, set_units="voxel")
