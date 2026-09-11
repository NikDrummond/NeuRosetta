"""Helpers for registry-driven metric invariance tests."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from neurosetta.api import Tree
from neurosetta.ops.tree_graphs.tree_transformations import (
    rotate_coordinates,
    scale_coordinates,
    translate_coordinates,
)
from neurosetta.utils.metrics.registry import METRIC_DEFINITIONS, MetricDefinition

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

TRANSLATION = (123.4, -57.2, 19.75)
SCALE = 2.5
# Deterministic Euler angles (radians): 31°, 47°, 23°.
_EULER_DEG = (31.0, 47.0, 23.0)
EULER_RAD = tuple(math.radians(a) for a in _EULER_DEG)

_REF_AXIS = (0.0, 0.0, 1.0)

_EDGE_CACHE = ("Path_length", "Euclidean_length", "Edge_angle", "Radial_angle")
_GRAPH_CACHE = ("Convex_hull",)


def clear_geometry_caches(tree: Tree, *, lengths: bool = True) -> None:
    """Drop cached geometry that can go stale after coordinate transforms."""
    g = tree.graph
    props = _EDGE_CACHE if lengths else ("Edge_angle", "Radial_angle")
    for name in props:
        if name in g.ep:
            del g.ep[name]
    for name in _GRAPH_CACHE:
        if name in g.gp:
            del g.gp[name]


def translated_tree(tree: Tree) -> Tree:
    out = tree.copy()
    translate_coordinates(out, *TRANSLATION, bind=True)
    clear_geometry_caches(out, lengths=False)
    return out


def rotated_tree(tree: Tree) -> Tree:
    """Apply a general 3D Euler rotation about the origin (NeuRosetta rotates)."""
    out = tree.copy()
    rx, ry, rz = EULER_RAD
    rotate_coordinates(out, (1.0, 0.0, 0.0), rx, assume_normalized=True, bind=True)
    rotate_coordinates(out, (0.0, 1.0, 0.0), ry, assume_normalized=True, bind=True)
    rotate_coordinates(out, (0.0, 0.0, 1.0), rz, assume_normalized=True, bind=True)
    clear_geometry_caches(out, lengths=False)
    return out


def scaled_tree(tree: Tree, factor: float = SCALE) -> Tree:
    """Uniform scale about the origin; refresh length caches from coordinates."""
    from neurosetta.ops.tree_graphs.tree_path_lengths import get_edge_length

    out = tree.copy()
    scale_coordinates(out, factor, center=(0.0, 0.0, 0.0), scale_radii=True, bind=True)
    clear_geometry_caches(out, lengths=True)
    get_edge_length(out, bind=True, recalculate=True)
    return out


# ---------------------------------------------------------------------------
# Exclusions — every registered metric must be tested or listed here
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Exclusion:
    name: str
    reason: str


EXCLUSIONS: tuple[Exclusion, ...] = (
    Exclusion(
        "tree_summary",
        "Returns SummaryTable mixing heterogeneous columns; not a single numeric output.",
    ),
    Exclusion(
        "summary_table",
        "Returns DataFrame mixing topology counts and cable length.",
    ),
    Exclusion(
        "forest_summary",
        "Forest-only display aggregate; covered by dedicated forest summary smoke if needed.",
    ),
    Exclusion(
        "forest_summary_table",
        "Forest-only DataFrame aggregate; not a Tree metric.",
    ),
    Exclusion(
        "get_convex_hull",
        "Returns SciPy ConvexHull with absolute vertices; volume tested via get_convex_hull_volume.",
    ),
    Exclusion(
        "fit_sphere",
        "Multi-output (center, radius) with mixed invariance; optimization-based.",
    ),
    Exclusion(
        "fit_line",
        "Multi-output (slope, center, variances) with mixed invariance and sign ambiguity.",
    ),
    Exclusion(
        "fit_plane",
        "Multi-output (normal, center, variance) with mixed invariance and sign ambiguity.",
    ),
    Exclusion(
        "fit_circle",
        "Multi-output (center, radius, normal) with mixed invariance and sign ambiguity.",
    ),
    Exclusion(
        "distance_from_neuropil_surface",
        "Requires external neuropil mesh; covered by dedicated mesh-relative tests.",
    ),
    Exclusion(
        "neuropil_point_depth",
        "Requires external neuropil mesh; covered by dedicated mesh-relative tests.",
    ),
    Exclusion(
        "coordinate_pca",
        "Returns (evals, evecs); evecs have sign ambiguity. Eigenvalue TI/RI covered by dedicated PCA tests.",
    ),
)

EXCLUDED_NAMES: frozenset[str] = frozenset(item.name for item in EXCLUSIONS)


def _call_kwargs(defn: MetricDefinition) -> dict[str, Any]:
    """Extra kwargs needed to evaluate a metric on the invariance fixture."""
    name = defn.name
    kwargs: dict[str, Any] = {}

    if name in {"get_edge_angles", "get_mean_edge_angle", "get_edge_angle_variance"}:
        kwargs["between_vector"] = _REF_AXIS
        if name == "get_mean_edge_angle":
            kwargs["perspective_vector"] = None
        kwargs["signed"] = False

    if name.startswith("coordinate_") and (
        name.endswith("_along_axis") or name == "coordinate_projection_moments"
    ):
        kwargs["axis"] = _REF_AXIS

    if name in {
        "get_edge_angles",
        "get_edge_length",
        "get_subtree_scores",
        "get_partition_asymmetry",
        "get_node_depth",
        "get_radial_angle",
    }:
        kwargs["bind"] = False

    if name == "get_radial_angle":
        kwargs["signed"] = False

    if name == "get_partition_asymmetry":
        # Weighted PA uses Path_length cable fractions (still scale-invariant).
        kwargs["weighted"] = True

    return kwargs


def invoke_metric(defn: MetricDefinition, tree: Tree) -> Any:
    """Evaluate a Tree-exposed metric for invariance comparison."""
    if defn.tree_method in (None, ""):
        raise ValueError(f"{defn.name} is not exposed on Tree")
    method = getattr(tree, defn.tree_method)
    return method(**_call_kwargs(defn))


def _as_numeric(value: Any) -> Any:
    """Convert metric outputs into comparable numeric structures."""
    if value is None:
        raise AssertionError("metric returned None; pass bind=False where applicable")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, pd.DataFrame):
        return value.to_numpy(dtype=float)
    if isinstance(value, tuple):
        return tuple(_as_numeric(v) for v in value)
    if isinstance(value, list):
        return [_as_numeric(v) for v in value]
    arr = np.asarray(value)
    if arr.dtype == object:
        return [_as_numeric(v) for v in value]
    if np.issubdtype(arr.dtype, np.number):
        return np.asarray(arr, dtype=float)
    raise TypeError(f"unsupported metric output type: {type(value)!r}")


def assert_metric_equal(a: Any, b: Any, *, rtol: float = 1e-9, atol: float = 1e-9) -> None:
    """Compare metric outputs with floating-point tolerance."""
    na = _as_numeric(a)
    nb = _as_numeric(b)
    if isinstance(na, (int, float)) and isinstance(nb, (int, float)):
        if isinstance(na, int) and isinstance(nb, int):
            assert na == nb
        else:
            assert_allclose(na, nb, rtol=rtol, atol=atol)
        return
    if isinstance(na, tuple) and isinstance(nb, tuple):
        assert len(na) == len(nb)
        for xa, xb in zip(na, nb, strict=True):
            assert_metric_equal(xa, xb, rtol=rtol, atol=atol)
        return
    if isinstance(na, list) and isinstance(nb, list):
        assert len(na) == len(nb)
        for xa, xb in zip(na, nb, strict=True):
            assert_metric_equal(xa, xb, rtol=rtol, atol=atol)
        return
    assert_allclose(np.asarray(na, dtype=float), np.asarray(nb, dtype=float), rtol=rtol, atol=atol)


def testable_definitions(
    *,
    attribute: str | None = None,
    attribute_value: bool = True,
) -> list[MetricDefinition]:
    """Registered metrics eligible for generic invariance tests."""
    out: list[MetricDefinition] = []
    for defn in METRIC_DEFINITIONS:
        if defn.name in EXCLUDED_NAMES:
            continue
        if defn.tree_method in (None, ""):
            continue
        if attribute is not None and getattr(defn, attribute) is not attribute_value:
            continue
        out.append(defn)
    return out


def metric_ids(definitions: list[MetricDefinition]) -> list[str]:
    return [defn.name for defn in definitions]


def coverage_status() -> dict[str, str]:
    """Map every registered metric name to tested/excluded/dedicated."""
    status = {defn.name: "uncovered" for defn in METRIC_DEFINITIONS}
    for item in EXCLUSIONS:
        status[item.name] = f"excluded: {item.reason}"
    for defn in testable_definitions():
        status[defn.name] = "generic"
    # Dedicated markers filled by tests module.
    return status
