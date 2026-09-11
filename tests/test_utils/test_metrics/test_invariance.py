"""Numerical checks that declared metric invariance metadata holds in practice."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from vedo import Mesh

from neurosetta.api import Forest
from neurosetta.core.mesh import _Mesh
from neurosetta.ops.neuropils.distances import (
    distance_from_neuropil_surface,
    neuropil_point_depth,
)
from neurosetta.ops.tree_graphs.tree_coordinates import get_node_coordinates
from neurosetta.ops.tree_graphs.tree_transformations import translate_coordinates
from neurosetta.utils.metrics.registry import METRIC_DEFINITIONS

from fixtures import make_invariance_tree, make_invariance_tree_alt_coords
from invariance_helpers import (
    EXCLUDED_NAMES,
    EXCLUSIONS,
    SCALE,
    TRANSLATION,
    assert_metric_equal,
    clear_geometry_caches,
    invoke_metric,
    metric_ids,
    rotated_tree,
    scaled_tree,
    testable_definitions,
    translated_tree,
)

# Dedicated coverage names (must appear in the coverage audit).
_DEDICATED = frozenset(
    {
        "get_total_cable_length",
        "get_convex_hull_volume",
        "get_bifurcation_angles",
        "coordinate_pca",
        "distance_from_neuropil_surface",
        "neuropil_point_depth",
        "forest_summary_table",
    }
)


@pytest.fixture
def inv_tree():
    return make_invariance_tree()


# ---------------------------------------------------------------------------
# Coverage: every registered metric is generic / dedicated / excluded
# ---------------------------------------------------------------------------


def test_every_registered_metric_is_covered():
    registered = {defn.name for defn in METRIC_DEFINITIONS}
    excluded = {item.name for item in EXCLUSIONS}
    generic = {defn.name for defn in testable_definitions()}

    assert excluded == EXCLUDED_NAMES
    assert excluded <= registered
    uncovered = registered - excluded - generic - _DEDICATED
    # Dedicated metrics may also be generic; that's fine.
    uncovered = registered - excluded - generic
    # Anything left must be dedicated-only (tree_method missing etc.)
    leftover = uncovered - _DEDICATED
    assert not leftover, f"metrics neither tested nor excluded: {sorted(leftover)}"

    # Every exclusion names a real metric.
    assert excluded <= registered


def test_exclusion_reasons_are_nonempty():
    assert EXCLUSIONS
    for item in EXCLUSIONS:
        assert item.reason.strip()


# ---------------------------------------------------------------------------
# Generic declared-True invariance
# ---------------------------------------------------------------------------

_TI = testable_definitions(attribute="translation_invariant", attribute_value=True)
_RI = testable_definitions(attribute="rotation_invariant", attribute_value=True)
_SI = testable_definitions(attribute="scale_invariant", attribute_value=True)
_TOPO = [
    defn
    for defn in testable_definitions()
    if defn.domain == "topology" and not defn.requires_coordinates
]


@pytest.mark.parametrize("defn", _TI, ids=metric_ids(_TI))
def test_declared_translation_invariance(defn, inv_tree):
    original = invoke_metric(defn, inv_tree)
    transformed = invoke_metric(defn, translated_tree(inv_tree))
    assert_metric_equal(original, transformed)


@pytest.mark.parametrize("defn", _RI, ids=metric_ids(_RI))
def test_declared_rotation_invariance(defn, inv_tree):
    original = invoke_metric(defn, inv_tree)
    transformed = invoke_metric(defn, rotated_tree(inv_tree))
    assert_metric_equal(original, transformed)


@pytest.mark.parametrize("defn", _SI, ids=metric_ids(_SI))
def test_declared_scale_invariance(defn, inv_tree):
    original = invoke_metric(defn, inv_tree)
    transformed = invoke_metric(defn, scaled_tree(inv_tree))
    assert_metric_equal(original, transformed)


@pytest.mark.parametrize("defn", _TOPO, ids=metric_ids(_TOPO))
def test_topology_invariant_under_coordinate_replacement(defn):
    tree_a = make_invariance_tree()
    tree_b = make_invariance_tree_alt_coords()
    assert tree_a.graph.num_edges() == tree_b.graph.num_edges()
    a = invoke_metric(defn, tree_a)
    b = invoke_metric(defn, tree_b)
    assert_metric_equal(a, b)


# ---------------------------------------------------------------------------
# Dedicated transformation behaviour
# ---------------------------------------------------------------------------


def test_cable_length_rigid_invariant_and_scale_covariant(inv_tree):
    defn = next(d for d in METRIC_DEFINITIONS if d.name == "get_total_cable_length")
    L0 = invoke_metric(defn, inv_tree)
    assert_metric_equal(L0, invoke_metric(defn, translated_tree(inv_tree)))
    assert_metric_equal(L0, invoke_metric(defn, rotated_tree(inv_tree)))
    Ls = invoke_metric(defn, scaled_tree(inv_tree))
    assert_allclose(Ls, SCALE * L0, rtol=1e-9, atol=1e-9)


def test_convex_hull_volume_rigid_invariant_and_scale_cubed(inv_tree):
    defn = next(d for d in METRIC_DEFINITIONS if d.name == "get_convex_hull_volume")
    V0 = invoke_metric(defn, inv_tree)
    assert_metric_equal(V0, invoke_metric(defn, translated_tree(inv_tree)))
    assert_metric_equal(V0, invoke_metric(defn, rotated_tree(inv_tree)))
    Vs = invoke_metric(defn, scaled_tree(inv_tree))
    assert_allclose(Vs, (SCALE**3) * V0, rtol=1e-9, atol=1e-9)


def test_bifurcation_angles_fully_similarity_invariant(inv_tree):
    defn = next(d for d in METRIC_DEFINITIONS if d.name == "get_bifurcation_angles")
    a0 = invoke_metric(defn, inv_tree)
    assert_metric_equal(a0, invoke_metric(defn, translated_tree(inv_tree)))
    assert_metric_equal(a0, invoke_metric(defn, rotated_tree(inv_tree)))
    assert_metric_equal(a0, invoke_metric(defn, scaled_tree(inv_tree)))


def test_pca_eigenvalues_translation_and_rotation_invariant(inv_tree):
    defn = next(d for d in METRIC_DEFINITIONS if d.name == "coordinate_pca")
    evals0, _ = invoke_metric(defn, inv_tree)
    evals_t, _ = invoke_metric(defn, translated_tree(inv_tree))
    evals_r, _ = invoke_metric(defn, rotated_tree(inv_tree))
    assert_allclose(evals0, evals_t, rtol=1e-9, atol=1e-9)
    assert_allclose(evals0, evals_r, rtol=1e-9, atol=1e-9)


def test_pca_eigenvalues_scale_invariant_when_normalized(inv_tree):
    # Classic (non-robust) covariance is exactly scale-invariant when normalised.
    evals0, _ = inv_tree.coordinate_pca(norm=True, robust=False)
    evals_s, _ = scaled_tree(inv_tree).coordinate_pca(norm=True, robust=False)
    assert_allclose(evals0, evals_s, rtol=1e-9, atol=1e-12)


def _unit_box_mesh() -> _Mesh:
    # Axis-aligned unit cube.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 3, 7],
            [0, 7, 4],
        ],
        dtype=int,
    )
    return _Mesh(ID="box", metadata={}, mesh=Mesh([verts, faces]))


def test_neuropil_distance_changes_when_only_points_move():
    mesh = _unit_box_mesh()
    points = np.array([[0.5, 0.5, 2.0], [0.5, 0.5, -1.0]], dtype=float)
    d0 = distance_from_neuropil_surface(mesh, points)
    d1 = distance_from_neuropil_surface(mesh, points + np.asarray(TRANSLATION))
    assert not np.allclose(d0, d1)


def test_neuropil_distance_rigid_invariant_when_points_and_mesh_transform_together():
    mesh = _unit_box_mesh()
    points = np.array([[0.5, 0.5, 2.0], [1.5, 0.5, 0.5]], dtype=float)
    d0 = distance_from_neuropil_surface(mesh, points)

    dx, dy, dz = TRANSLATION
    moved_points = points + np.asarray(TRANSLATION)
    moved_mesh = mesh.copy()
    moved_mesh.mesh.shift(dx, dy, dz)
    d1 = distance_from_neuropil_surface(moved_mesh, moved_points)
    assert_allclose(d0, d1, rtol=1e-5, atol=1e-5)


def test_neuropil_point_depth_runs_on_simple_mesh():
    # Two coplanar triangles with opposite winding (same pattern as surface_distance tests).
    verts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float)
    faces = np.array([[0, 1, 2], [0, 2, 1]], dtype=int)
    mesh = _Mesh(ID="planes", metadata={}, mesh=Mesh([verts, faces]))
    points = np.array([[0.25, 0.25, 0.5], [0.5, 0.1, 0.5]], dtype=float)
    depth = neuropil_point_depth(mesh, points, t=0.0, surface="outer", norm=True)
    assert depth.shape == (2,)
    assert np.all(np.isfinite(depth))


def test_forest_summary_table_stable_under_shared_translation():
    trees = [make_invariance_tree(units="micron") for _ in range(2)]
    # Offset second tree so the forest is nontrivial.
    translate_coordinates(trees[1], 5.0, 0.0, 0.0, bind=True)
    clear_geometry_caches(trees[1], lengths=False)
    forest = Forest([t.copy() for t in trees])
    moved = Forest([translated_tree(t) for t in trees])
    a = forest.summary_table()
    b = moved.summary_table()
    # Topology columns identical; cable lengths unchanged under translation.
    assert_allclose(a[["nodes", "branches", "leaves", "cable"]].to_numpy(dtype=float),
                    b[["nodes", "branches", "leaves", "cable"]].to_numpy(dtype=float),
                    rtol=1e-9, atol=1e-9)


def test_fixture_is_nonplanar_asymmetric_and_branched(inv_tree):
    coords = get_node_coordinates(inv_tree, SoA=False)
    assert coords.shape[0] >= 10
    assert np.linalg.matrix_rank(coords - coords.mean(axis=0), tol=1e-8) == 3
    assert inv_tree.count_bifurcations() >= 2
    lengths = inv_tree.get_edge_length(bind=False, recalculate=True)
    assert np.all(lengths > 0)
    assert len(np.unique(np.round(lengths, 6))) > 1
    evals, _ = inv_tree.coordinate_pca(norm=True)
    gaps = np.diff(np.sort(evals))
    assert np.min(np.abs(gaps)) > 1e-6


# ---------------------------------------------------------------------------
# Counts for the deliverable (also useful as regression anchors)
# ---------------------------------------------------------------------------


def test_invariance_suite_size_anchors():
    assert len(_TI) >= 20
    assert len(_RI) >= 15
    assert len(_SI) >= 15
    assert len(_TOPO) == 20
