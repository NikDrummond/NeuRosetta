"""Tree-level synapse API tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurosetta.api import Tree


@pytest.fixture
def tree(test_tree) -> Tree:
    return Tree(ID=test_tree.ID, metadata=dict(test_tree.metadata), graph=test_tree.graph)


@pytest.fixture
def synapse_df(tree: Tree) -> pd.DataFrame:
    starts, ends = tree.get_edge_coordinates()
    # On first edge midpoint + one offset postsynaptic point.
    mid = 0.5 * (starts[0] + ends[0])
    beside = mid + np.array([0.0, 0.5, 0.0])
    return pd.DataFrame(
        {
            "synapse_id": [10, 11, 12],
            "type": ["pre", "post", "output"],
            "x": [mid[0], beside[0], mid[0] + 0.1],
            "y": [mid[1], beside[1], mid[1]],
            "z": [mid[2], beside[2], mid[2]],
            "partner_id": [100, 200, 100],
            "confidence": [0.9, 0.8, 0.7],
        }
    )


def test_set_and_filter(tree, synapse_df):
    tree.set_synapses(synapse_df)
    assert tree.synapses is not None
    assert len(tree.synapses) == 3
    assert len(tree.synapses.pre) == 2  # pre + output alias
    assert len(tree.synapses.post) == 1
    assert len(tree.synapses.inputs) == 1
    assert len(tree.synapses.outputs) == 2
    assert len(tree.synapses.filter(partner_id=100)) == 2
    assert len(tree.synapses.filter(partner_id=[100, 200])) == 3
    assert "confidence" in tree.synapses.columns


def test_mapping_and_path_distance(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    syn = tree.synapses
    assert syn.is_mapped
    assert syn.mapping_valid
    df = syn.to_dataframe(copy=False)
    assert set(df["edge_index"].tolist())  # non-empty
    assert np.all((df["edge_fraction"] >= 0) & (df["edge_fraction"] <= 1))
    # Synapse 0 sits on edge 0 midpoint.
    row0 = df.loc[df["synapse_id"] == 10].iloc[0]
    assert int(row0["edge_index"]) == 0
    assert row0["edge_fraction"] == pytest.approx(0.5, abs=1e-5)
    assert row0["distance_to_tree"] == pytest.approx(0.0, abs=1e-5)

    path = tree.get_synapse_path_distance()
    assert path.shape == (3,)
    assert np.isfinite(path[0])


def test_max_distance_marks_unmapped(tree, synapse_df):
    far = synapse_df.copy()
    far.loc[1, ["x", "y", "z"]] = [1e6, 1e6, 1e6]
    tree.set_synapses(far)
    tree.map_synapses(max_distance=0.1)
    summary = tree.synapse_mapping_summary()
    assert summary["n_synapses"] == 3
    assert summary["n_unmapped"] >= 1
    assert summary["n_mapped"] + summary["n_unmapped"] == 3


def test_drop_unmapped(tree, synapse_df):
    far = synapse_df.copy()
    far.loc[1, ["x", "y", "z"]] = [1e6, 1e6, 1e6]
    tree.set_synapses(far)
    tree.map_synapses(max_distance=0.1, drop_unmapped=True)
    assert tree.synapses is not None
    assert len(tree.synapses) == 2
    assert int(tree.synapses.to_dataframe(copy=False)["mapped"].sum()) == 2


def test_counts_density_edge_counts(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    assert tree.count_synapses(type="both") == 3
    assert tree.count_synapses(type="pre") == 2
    assert tree.count_synapses(type="post") == 1
    dens = tree.synapse_density(type="both")
    assert dens > 0
    counts = tree.get_edge_synapse_counts(type="both")
    assert counts.shape[0] == tree.count_edges()
    assert counts.sum() == 3


def test_translate_keeps_mapping(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    before = tree.synapses.to_dataframe(copy=True)
    tree.translate_coordinates(10.0, -5.0, 2.0)
    after = tree.synapses.to_dataframe(copy=False)
    np.testing.assert_allclose(after["x"], before["x"] + 10.0)
    np.testing.assert_allclose(after["nearest_x"], before["nearest_x"] + 10.0)
    np.testing.assert_array_equal(after["edge_index"], before["edge_index"])
    assert tree.synapses.mapping_valid


def test_scale_scales_distances(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    before = tree.synapses.to_dataframe(copy=True)["distance_to_tree"].to_numpy()
    tree.scale_coordinates(2.0, center=(0, 0, 0))
    after = tree.synapses.to_dataframe(copy=False)["distance_to_tree"].to_numpy()
    np.testing.assert_allclose(after, before * 2.0)


def test_unit_conversion_scales_synapses(tree, synapse_df):
    tree.set_units("nm")
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    x0 = tree.synapses.coordinates[0, 0]
    tree.convert_units("um")
    assert tree.synapses.coordinates[0, 0] == pytest.approx(x0 / 1000.0)


def test_reroot_invalidates_mapping(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    assert tree.synapses.mapping_valid
    leaves = tree.get_leaf_indices()
    tree.get_rerooted_tree(int(leaves[0]), inplace=True)
    assert not tree.synapses.mapping_valid


def test_reduce_transfers_mapping(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    n = len(tree.synapses)
    before_path = tree.get_synapse_path_distance().copy()
    tree.get_reduced_tree(inplace=True)
    assert tree.synapses is not None
    assert len(tree.synapses) == n
    assert tree.synapses.mapping_valid
    after_path = tree.get_synapse_path_distance()
    np.testing.assert_allclose(before_path, after_path, atol=1e-5)


def test_copy_preserves_synapses(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    clone = tree.copy()
    assert clone.synapses is not None
    assert len(clone.synapses) == 3
    assert clone.synapses.is_mapped


def test_nr_roundtrip(tree, synapse_df, tmp_path):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    path = tmp_path / "1.nr"
    tree.save_tree(path)
    from neurosetta.io import load

    loaded = load(path)
    assert loaded.synapses is not None
    assert len(loaded.synapses) == 3
    assert loaded.synapses.is_mapped
    assert "confidence" in loaded.synapses.columns


def test_plot_2d_with_synapses(tree, synapse_df):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    ax = tree.show_2d(synapses="both", synapse_position="mapped", show_synapse_mapping=True)
    assert ax is not None


def test_plot_show_synapses_alias_and_colour_by(tree, synapse_df):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    df = synapse_df.copy()
    df["partner_type"] = ["T4", "T5", "T4"]
    tree.set_synapses(df)
    tree.map_synapses()
    ax = tree.show_2d(show_synapses=True, synapse_colour_by="partner_type")
    assert ax is not None
    ax2 = tree.show_2d(synapses="input")  # alias → post
    assert ax2 is not None


def test_plot_3d_actors(tree, synapse_df):
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    from neurosetta.ops.plotting import Viewer

    viewer = Viewer(offscreen=True)
    viewer.add_neuron(tree)
    actors = viewer.add_synapses(tree, show_synapses=True, show_synapse_mapping=True)
    assert len(actors) >= 2
    df = synapse_df.copy()
    df["nt"] = ["ACh", "Glu", "ACh"]
    tree.set_synapses(df)
    tree.map_synapses()
    coloured = viewer.add_synapses(tree, synapses="both", synapse_colour_by="nt")
    assert len(coloured) >= 1
    viewer.close()


def test_subtree_keeps_mapped_synapses_only(tree, synapse_df):
    starts, ends = tree.get_edge_coordinates()
    # Put one synapse on edge 0, one on the last edge, one far away.
    mid0 = 0.5 * (starts[0] + ends[0])
    mid_last = 0.5 * (starts[-1] + ends[-1])
    df = pd.DataFrame(
        {
            "synapse_id": [1, 2, 3],
            "type": ["pre", "post", "pre"],
            "x": [mid0[0], mid_last[0], 1e6],
            "y": [mid0[1], mid_last[1], 1e6],
            "z": [mid0[2], mid_last[2], 1e6],
            "partner_id": [10, 20, 30],
        }
    )
    tree.set_synapses(df)
    tree.map_synapses(max_distance=1.0)
    assert tree.count_synapses() == 3
    assert tree.synapse_mapping_summary()["n_mapped"] == 2

    # Mask a small subtree rooted at the child of edge 0's target if possible,
    # otherwise at a leaf — then extract with default mapped_only policy.
    edges = tree.get_edge_indices()
    # Root subtree at the target of edge 0 so edge 0 is typically excluded
    # (subtree edges are descendants of the new root).
    child = int(edges[0, 1])
    tree.subtree_mask_from_root(child)
    before_nodes = tree.count_nodes()
    tree.get_subtree(synapses="mapped_only")
    assert tree.count_nodes() < before_nodes
    # Far + edge-0 synapses should be gone; last-edge synapse only kept if that
    # edge was retained.
    syn = tree.synapses
    if syn is not None:
        assert syn.is_mapped
        assert bool(syn.to_dataframe(copy=False)["mapped"].all())
        assert 1 not in set(syn.to_dataframe(copy=False)["synapse_id"])
        assert 3 not in set(syn.to_dataframe(copy=False)["synapse_id"])


def test_subtree_unmapped_cleared_by_default(tree, synapse_df):
    tree.set_synapses(synapse_df)
    # Never map — default policy clears.
    tree.subtree_mask_from_root(int(tree.get_leaf_indices()[0]))
    tree.get_subtree(synapses="mapped_only")
    assert tree.synapses is None or len(tree.synapses) == 0


def test_subtree_all_keeps_unmapped_table(tree, synapse_df):
    tree.set_synapses(synapse_df)
    n = len(tree.synapses)
    tree.subtree_mask_from_root(int(tree.get_leaf_indices()[0]))
    tree.get_subtree(synapses="all")
    assert tree.synapses is not None
    assert len(tree.synapses) == n
    assert not tree.synapses.mapping_valid
