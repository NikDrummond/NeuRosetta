"""Tests for ``neurosetta.describe``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fixtures import make_invariance_tree
from vedo import Mesh

from neurosetta import describe
from neurosetta.core.mesh import _Mesh
from neurosetta.testing import make_synthetic_forest
from neurosetta.utils.metrics.descriptors import select_describe_definitions


@pytest.fixture
def tree():
    return make_invariance_tree(units="micron")


@pytest.fixture
def forest():
    return make_synthetic_forest(n_trees=3, n=25, seed=3, units="micron")


def test_describe_tree_wide_one_row(tree):
    df = describe(tree, parallel=False)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.loc[0, "neuron_id"] == tree.ID
    assert "topology.max_depth" in df.columns
    assert "intrinsic_geometry.total_cable_length" in df.columns
    # No embedding without reference_frame.
    assert not any(c.startswith("embedding.") for c in df.columns)


def test_describe_forest_one_row_per_tree(forest):
    df = describe(forest, parallel=False)
    assert len(df) == len(forest)
    assert list(df["neuron_id"]) == [t.ID for t in forest]


def test_describe_domain_topology_only(tree):
    df = describe(tree, domains="topology", parallel=False)
    assert all(c == "neuron_id" or c.startswith("topology.") for c in df.columns)
    assert not any("intrinsic_geometry." in c for c in df.columns)
    assert not any(c.startswith("embedding.") for c in df.columns)


def test_describe_explicit_metrics_intersection(tree):
    df = describe(
        tree,
        domains="topology",
        metrics=["get_max_depth", "get_total_cable_length"],
        parallel=False,
    )
    # Intersection: cable length is intrinsic_geometry → dropped.
    assert "topology.max_depth" in df.columns
    assert "intrinsic_geometry.total_cable_length" not in df.columns


def test_describe_explicit_metrics_only(tree):
    df = describe(
        tree,
        metrics=["get_max_depth", "get_total_cable_length"],
        parallel=False,
    )
    data_cols = [c for c in df.columns if c != "neuron_id"]
    assert data_cols == [
        "topology.max_depth",
        "intrinsic_geometry.total_cable_length",
    ]


def test_bifurcation_angle_summaries_match_numpy(tree):
    df = describe(
        tree,
        metrics=["get_bifurcation_angles"],
        summaries=["mean", "std", "median"],
        parallel=False,
    )
    pc1, pc2, c = tree.get_bifurcation_angles()
    arr = np.concatenate([np.ravel(pc1), np.ravel(pc2), np.ravel(c)])
    assert df.loc[0, "intrinsic_geometry.bifurcation_angles.mean"] == pytest.approx(
        float(np.mean(arr))
    )
    assert df.loc[0, "intrinsic_geometry.bifurcation_angles.std"] == pytest.approx(
        float(np.std(arr))
    )
    assert df.loc[0, "intrinsic_geometry.bifurcation_angles.median"] == pytest.approx(
        float(np.median(arr))
    )


def test_wide_long_equivalence(tree):
    wide = describe(tree, parallel=False)
    long = describe(tree, output="long", parallel=False)
    for _, row in long.iterrows():
        summary = row["summary"]
        if summary is None or (isinstance(summary, float) and np.isnan(summary)):
            col = f"{row['domain']}.{row['metric']}"
        else:
            col = f"{row['domain']}.{row['metric']}.{summary}"
        assert col in wide.columns
        assert wide.loc[0, col] == pytest.approx(row["value"], nan_ok=True)


def test_long_units_column(tree):
    long = describe(
        tree,
        metrics=["get_total_cable_length", "get_max_depth"],
        output="long",
        parallel=False,
    )
    cable = long.loc[long["metric"] == "total_cable_length"].iloc[0]
    depth = long.loc[long["metric"] == "max_depth"].iloc[0]
    assert cable["unit"] == "micron"
    assert depth["unit"] == "dimensionless"


def test_missing_reference_frame_explicit_metric_raises(tree):
    with pytest.raises(ValueError, match="reference_frame"):
        describe(tree, metrics=["coordinate_mean_along_axis"], parallel=False)


def test_embedding_with_axis_reference(tree):
    df = describe(
        tree,
        domains="embedding",
        reference_frame=(0.0, 0.0, 1.0),
        metrics=["coordinate_extent_along_axis"],
        parallel=False,
    )
    assert "embedding.coordinate_extent_along_axis" in df.columns
    assert df.loc[0, "embedding.coordinate_extent_along_axis"] == pytest.approx(
        tree.coordinate_extent_along_axis((0.0, 0.0, 1.0))
    )


def test_unsupported_metric_explicit_raises(tree):
    with pytest.raises(ValueError, match="unsupported"):
        describe(tree, metrics=["fit_sphere"], parallel=False)


def test_errors_warn_fills_nan(tree, monkeypatch):
    import neurosetta.utils.metrics.descriptors as descriptors_mod

    def boom(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(descriptors_mod, "_invoke_raw", boom)
    with pytest.warns(UserWarning, match="boom"):
        df = describe(
            tree,
            metrics=["get_max_depth"],
            errors="warn",
            parallel=False,
        )
    assert np.isnan(df.loc[0, "topology.max_depth"])


def test_describe_ordering_stable(tree):
    a = list(describe(tree, parallel=False).columns)
    b = list(describe(tree, parallel=False).columns)
    assert a == b


def test_empty_distribution_summaries_are_nan_with_zero_count():
    # Single-edge line: no bifurcations.
    from graph_tool.all import Graph

    from neurosetta.api import Tree
    from neurosetta.utils.graph_utils.node_types import infer_node_types

    g = Graph(directed=True)
    g.add_vertex(2)
    g.add_edge(0, 1)
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2]])
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    g.vp["radius"] = g.new_vp("double", np.ones(2) * 0.5)
    g.vp["node_type"] = g.new_vp("int", infer_node_types(g))
    tree = Tree(ID=99, metadata={"units": "micron", "isReduced": False}, graph=g)

    df = describe(
        tree,
        metrics=["get_bifurcation_angles"],
        summaries=["count", "mean"],
        parallel=False,
    )
    assert df.loc[0, "intrinsic_geometry.bifurcation_angles.count"] == 0.0
    assert np.isnan(df.loc[0, "intrinsic_geometry.bifurcation_angles.mean"])


def test_include_metadata_units(forest):
    df = describe(forest, include_metadata=True, parallel=False)
    assert "meta.units" in df.columns
    assert list(df["meta.units"]) == ["micron"] * len(forest)


def test_select_definitions_default_excludes_embedding():
    selected = select_describe_definitions()
    assert selected
    assert all(d.domain in {"topology", "intrinsic_geometry"} for d in selected)
    assert all(not d.requires_reference_frame for d in selected)


def test_mesh_embedding_summarizes_distances(tree):
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    mesh = _Mesh(ID="tri", metadata={}, mesh=Mesh([verts, faces]))
    df = describe(
        tree,
        metrics=["distance_from_neuropil_surface"],
        reference_frame=mesh,
        summaries=["mean", "count"],
        parallel=False,
    )
    assert "embedding.distance_from_neuropil_surface.mean" in df.columns
    assert df.loc[0, "embedding.distance_from_neuropil_surface.count"] == tree.count_nodes()
