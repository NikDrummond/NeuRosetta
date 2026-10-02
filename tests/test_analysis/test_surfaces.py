"""Tests for neuropil surface reconstruction."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from vedo import Mesh as VedoMesh

from neurosetta import Neuropil, reconstruct_neuropil_surface
from neurosetta.analysis.surfaces import (
    clean_mesh,
    generate_voxel_grid,
    reconstruct_surface_voxel,
    surface_from_voxel_grid,
)
from neurosetta.core.mesh import is_neuropil_mesh
from neurosetta.testing import make_synthetic_forest
from neurosetta.utils.units import normalize_units_str


@pytest.fixture
def micron_forest():
    return make_synthetic_forest(
        n_trees=3,
        n=40,
        seed=0,
        tree_spacing=8.0,
        units="micron",
    )


def _small_cloud(n: int = 120, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * 4.0


def test_clean_mesh_returns_trimesh():
    pts = _small_cloud()
    grid, offset, vs = generate_voxel_grid(pts, voxel_size=1.5, padding=1, dilate_iter=1)
    raw = surface_from_voxel_grid(grid, offset, vs)
    cleaned = clean_mesh(raw, voxel_size=1.5)
    assert isinstance(cleaned, trimesh.Trimesh)
    assert len(cleaned.vertices) > 0
    assert len(cleaned.faces) > 0
    assert cleaned.volume >= 0


def test_clean_mesh_accepts_vedo_mesh():
    pts = _small_cloud(80)
    grid, offset, vs = generate_voxel_grid(pts, voxel_size=1.5, padding=1, dilate_iter=1)
    raw = surface_from_voxel_grid(grid, offset, vs)
    vedo = VedoMesh([np.asarray(raw.vertices), np.asarray(raw.faces)])
    cleaned = clean_mesh(vedo, voxel_size=1.5)
    assert isinstance(cleaned, trimesh.Trimesh)
    assert len(cleaned.vertices) > 0


def test_reconstruct_surface_voxel_clean_flag():
    pts = _small_cloud()
    dirty = reconstruct_surface_voxel(pts, voxel_size=1.5, padding=1, dilate=1, clean=False)
    cleaned = reconstruct_surface_voxel(pts, voxel_size=1.5, padding=1, dilate=1, clean=True)
    assert isinstance(dirty, trimesh.Trimesh)
    assert isinstance(cleaned, trimesh.Trimesh)
    assert len(cleaned.vertices) > 0


def test_reconstruct_neuropil_surface(micron_forest):
    npil = reconstruct_neuropil_surface(
        micron_forest,
        name="test_np",
        voxel_size=3,
        remove_outliers=False,
        smooth=False,
        clean=True,
    )
    assert isinstance(npil, Neuropil)
    assert npil.ID == "test_np"
    assert is_neuropil_mesh(npil)
    assert normalize_units_str(npil.metadata["units"]) == normalize_units_str("micron")
    assert npil.mesh.npoints > 0
    assert len(np.asarray(npil.mesh.cells)) > 0


def test_reconstruct_neuropil_without_clean(micron_forest):
    npil = reconstruct_neuropil_surface(
        micron_forest,
        name="raw",
        voxel_size=3,
        remove_outliers=False,
        smooth=False,
        clean=False,
    )
    assert isinstance(npil, Neuropil)
    assert npil.mesh.npoints > 0


def test_reconstruct_requires_units():
    forest = make_synthetic_forest(n_trees=2, n=30, seed=1, units=None)
    with (
        pytest.warns(UserWarning, match="dimensionless units"),
        pytest.raises(ValueError, match="dimensionless"),
    ):
        reconstruct_neuropil_surface(
            forest,
            name="bad",
            voxel_size=3,
            remove_outliers=False,
            smooth=False,
            clean=False,
        )
