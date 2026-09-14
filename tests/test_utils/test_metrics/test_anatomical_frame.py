"""Tests for AnatomicalFrame and embedding describe() integration."""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import make_invariance_tree
from invariance_helpers import TRANSLATION
from numpy.testing import assert_allclose
from vedo import Mesh

import neurosetta as nr
from neurosetta.api.anatomical_frame import FrameRequirementError
from neurosetta.core.mesh import _Mesh
from neurosetta.testing import make_synthetic_forest


def _triangle_mesh(offset=(0.0, 0.0, 0.0)) -> _Mesh:
    o = np.asarray(offset, dtype=float)
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float) + o
    faces = np.array([[0, 1, 2]], dtype=int)
    return _Mesh(ID="tri", metadata={"units": "micron"}, mesh=Mesh([verts, faces]))


def _depth_mesh() -> _Mesh:
    """Coplanar opposite-winding triangles (existing package depth fixture pattern)."""
    verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=int)
    return _Mesh(ID="planes", metadata={"units": "micron"}, mesh=Mesh([verts, faces]))


@pytest.fixture
def tree():
    return make_invariance_tree(units="micron")


def test_construction_stores_references():
    mesh = _triangle_mesh()
    frame = nr.AnatomicalFrame(
        name="lobula_plate",
        reference_mesh=mesh,
        axes={"depth": (0, 0, 2), "anteroposterior": (1, 0, 0)},
        metadata={"units": "micron", "source": "test"},
    )
    assert frame.name == "lobula_plate"
    assert frame.reference_mesh is mesh
    assert_allclose(frame.axes["depth"], [0, 0, 1])
    assert frame.default_axis == "depth"
    assert "depth" in repr(frame)


def test_optional_components_mesh_only_surfaces_only_axes_only():
    mesh = _triangle_mesh()
    inner = _triangle_mesh((0, 0, -1))
    outer = _triangle_mesh((0, 0, 1))

    m = nr.AnatomicalFrame(reference_mesh=mesh)
    assert m.has("reference_mesh") and not m.has("axis")

    s = nr.AnatomicalFrame(inner_surface=inner, outer_surface=outer)
    assert s.has("inner_surface") and s.has("outer_surface")

    a = nr.AnatomicalFrame(axes={"dv": (0, 1, 0)})
    assert a.has("axis") and a.default_axis == "dv"


def test_inner_without_outer_rejected():
    with pytest.raises(ValueError, match="together"):
        nr.AnatomicalFrame(inner_surface=_triangle_mesh())


def test_invalid_axes_rejected():
    with pytest.raises(ValueError, match="zero"):
        nr.AnatomicalFrame(axes={"depth": (0, 0, 0)})
    with pytest.raises(ValueError, match="finite"):
        nr.AnatomicalFrame(axes={"depth": (np.nan, 0, 1)})


def test_require_missing_component():
    frame = nr.AnatomicalFrame(name="x", axes={"depth": (0, 0, 1)})
    with pytest.raises(FrameRequirementError, match="reference_mesh") as exc:
        frame.require("reference_mesh", metric="distance_from_neuropil_surface")
    assert exc.value.metric == "distance_from_neuropil_surface"
    assert "reference_mesh" in exc.value.missing


def test_surface_distance_equivalence(tree):
    mesh = _triangle_mesh()
    points = tree.get_node_coordinates()
    frame = nr.AnatomicalFrame(name="np", reference_mesh=mesh, metadata={"units": "micron"})
    assert_allclose(
        frame.surface_distance(points),
        nr.distance_from_neuropil_surface(mesh, points),
    )


def test_normalized_depth_equivalence_single_mesh():
    mesh = _depth_mesh()
    points = np.array([[0.25, 0.25, 0.5], [0.5, 0.1, 0.5]], dtype=float)
    frame = nr.AnatomicalFrame(name="np", reference_mesh=mesh, depth_t=0.0, depth_surface="outer")
    assert_allclose(
        frame.normalized_depth(points),
        nr.neuropil_point_depth(mesh, points, t=0.0, surface="outer", norm=True),
    )


def test_normalized_depth_via_surface_pair():
    inner = _triangle_mesh((0, 0, -1))
    outer = _triangle_mesh((0, 0, 1))
    frame = nr.AnatomicalFrame(inner_surface=inner, outer_surface=outer)
    points = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]])
    depth = frame.normalized_depth(points, surface="inner", norm=True)
    assert_allclose(depth, [0.5, 0.5], atol=1e-6)


def test_describe_embedding_with_frame(tree):
    frame = nr.AnatomicalFrame(
        name="np",
        reference_mesh=_depth_mesh(),
        axes={"depth": (0, 0, 1)},
        metadata={"units": "micron"},
        depth_surface="outer",
    )
    df = nr.describe(tree, domains="embedding", reference_frame=frame, parallel=False)
    assert any(c.startswith("embedding.") for c in df.columns)
    assert "embedding.coordinate_extent_along_axis" in df.columns
    assert "embedding.distance_from_neuropil_surface.mean" in df.columns
    assert "embedding.neuropil_point_depth.mean" in df.columns


def test_describe_embedding_without_frame_raises(tree):
    with pytest.raises(ValueError, match="embedding"):
        nr.describe(tree, domains="embedding", parallel=False)


def test_describe_legacy_axis_still_works(tree):
    df = nr.describe(
        tree,
        domains="embedding",
        reference_frame=(0.0, 0.0, 1.0),
        metrics=["coordinate_extent_along_axis"],
        parallel=False,
    )
    assert df.loc[0, "embedding.coordinate_extent_along_axis"] == pytest.approx(
        tree.coordinate_extent_along_axis((0.0, 0.0, 1.0))
    )


def test_frame_describe_convenience(tree):
    frame = nr.AnatomicalFrame(axes={"depth": (0, 0, 1)})
    df = frame.describe(tree, metrics=["coordinate_extent_along_axis"], parallel=False)
    assert len(df) == 1


def test_shared_rigid_transform_preserves_surface_distance():
    mesh = _triangle_mesh()
    points = np.array([[0.5, 0.5, 2.0], [1.5, 0.5, 0.5]], dtype=float)
    d0 = nr.distance_from_neuropil_surface(mesh, points)

    dx, dy, dz = TRANSLATION
    moved_points = points + np.asarray(TRANSLATION)
    moved_mesh = mesh.copy()
    moved_mesh.mesh.shift(dx, dy, dz)
    d1 = nr.distance_from_neuropil_surface(moved_mesh, moved_points)
    assert_allclose(d0, d1, rtol=1e-5, atol=1e-5)

    frame0 = nr.AnatomicalFrame(reference_mesh=mesh)
    frame1 = nr.AnatomicalFrame(reference_mesh=moved_mesh)
    assert_allclose(
        frame0.surface_distance(points),
        frame1.surface_distance(moved_points),
        rtol=1e-5,
        atol=1e-5,
    )


def test_existing_low_level_apis_unchanged(tree):
    mesh = _triangle_mesh()
    pts = tree.get_node_coordinates()
    d = nr.distance_from_neuropil_surface(mesh, pts)
    assert d.shape == (tree.count_nodes(),)
    depth = nr.neuropil_point_depth(_depth_mesh(), pts, t=0.0, surface="outer", norm=True)
    assert depth.shape == (tree.count_nodes(),)
    assert np.all(np.isfinite(depth))


def test_angle_to_axis():
    frame = nr.AnatomicalFrame(axes={"depth": (0, 0, 1)})
    assert frame.angle_to_axis((0, 0, 2)) == pytest.approx(0.0)
    assert frame.angle_to_axis((1, 0, 0), degrees=True) == pytest.approx(90.0)


def test_units_mismatch_raises(tree):
    frame = nr.AnatomicalFrame(
        axes={"depth": (0, 0, 1)},
        metadata={"units": "nanometer"},
    )
    with pytest.raises(ValueError, match="incompatible"):
        nr.describe(
            tree,
            domains="embedding",
            reference_frame=frame,
            metrics=["coordinate_extent_along_axis"],
            parallel=False,
        )


def test_forest_shared_frame():
    forest = make_synthetic_forest(n_trees=2, n=15, seed=2, units="micron")
    frame = nr.AnatomicalFrame(axes={"depth": (0, 0, 1)}, metadata={"units": "micron"})
    df = nr.describe(
        forest,
        domains="embedding",
        reference_frame=frame,
        metrics=["coordinate_extent_along_axis"],
        parallel=False,
    )
    assert len(df) == 2
