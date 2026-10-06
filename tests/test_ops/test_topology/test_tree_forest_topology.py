"""Tree / Forest TMD API tests."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurosetta.api import Tree
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree


def _tree_with_root_distance(n: int = 30, seed: int = 0) -> Tree:
    tree = make_synthetic_tree(n=n, seed=seed)
    tree.get_root_distance(bind=True)
    return tree


def test_tree_tmd_and_diagram():
    tree = _tree_with_root_distance()
    result = tree.tmd(func="Root_distance", bind=False)
    assert result is not None
    assert "birth_ind" in result
    diag = tree.persistence_diagram(func="Root_distance", normalize=True)
    assert diag.shape[1] == 2
    assert np.all(diag[:, 0] <= diag[:, 1])


def test_tree_tmd_bind_cache():
    tree = _tree_with_root_distance()
    assert tree.tmd(func="Root_distance", bind=True) is None
    assert "TMD" in tree.graph.gp
    # cached
    again = tree.tmd(func="Root_distance", bind=False, recalculate=False)
    assert again is not None
    assert again["function"] == "Root_distance"


def test_tree_topology_distance_self_zero():
    tree = _tree_with_root_distance(seed=1)
    other = tree  # same object
    d = tree.topology_distance(other, func="Root_distance", metric="barcode")
    assert d == pytest.approx(0.0)


@pytest.mark.skipif(
    importlib.util.find_spec("gudhi") is None,
    reason="GUDHI not installed",
)
def test_forest_distance_matrix_and_images():
    forest = make_synthetic_forest(n_trees=3, n=40, seed=7)
    for t in forest:
        t.get_root_distance(bind=True)

    mat = forest.topology_distance_matrix(func="Root_distance", metric="bottleneck")
    assert mat.shape == (3, 3)
    np.testing.assert_allclose(np.diag(mat), 0.0, atol=1e-12)
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)

    images, transformer = forest.persistence_images(
        func="Root_distance",
        resolution=(5, 5),
    )
    assert images.shape == (3, 25)
    assert transformer is not None

    # ordering matches Forest order
    diags = forest.persistence_diagrams(func="Root_distance")
    assert len(diags) == len(forest)


def test_forest_tmds_ordering():
    forest = make_synthetic_forest(n_trees=3, n=25, seed=3)
    for t in forest:
        t.get_root_distance(bind=True)
    results = forest.tmds(func="Root_distance", bind=False)
    assert len(results) == 3
    # death indices are valid vertex ids
    for tree, res in zip(forest, results, strict=True):
        assert res["death_ind"].max() < tree.graph.num_vertices()


def test_describe_includes_tmd_metrics():
    from neurosetta import describe

    tree = _tree_with_root_distance(seed=5)
    df = describe(
        tree,
        metrics=["count_tmd_bars", "get_tmd_survival_lengths"],
        summaries=["count", "mean"],
        parallel=False,
    )
    assert "intrinsic_geometry.count_tmd_bars" in df.columns
    assert "intrinsic_geometry.tmd_survival_lengths.count" in df.columns
    assert "intrinsic_geometry.tmd_survival_lengths.mean" in df.columns
    assert df.loc[0, "intrinsic_geometry.count_tmd_bars"] == tree.count_tmd_bars()
