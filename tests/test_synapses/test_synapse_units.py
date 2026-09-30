"""Synapse spatial-unit binding / mapping checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurosetta import Synapses
from neurosetta.api import Tree


@pytest.fixture
def tree(test_tree) -> Tree:
    # Leave dimensionless so unset-unit warnings can be tested explicitly.
    return Tree(ID=test_tree.ID, metadata=dict(test_tree.metadata), graph=test_tree.graph)


@pytest.fixture
def synapse_df(tree: Tree) -> pd.DataFrame:
    starts, ends = tree.get_edge_coordinates()
    mid = 0.5 * (starts[0] + ends[0])
    return pd.DataFrame(
        {
            "synapse_id": [1, 2],
            "type": ["pre", "post"],
            "x": [mid[0], mid[0]],
            "y": [mid[1], mid[1]],
            "z": [mid[2], mid[2]],
            "partner_id": [10, 20],
        }
    )


def test_bind_warns_when_units_unset(tree, synapse_df):
    syn = Synapses(synapse_df)
    with pytest.warns(UserWarning, match="lack spatial units|has no units|dimensionless"):
        tree.set_synapses(syn)
    # Tree was dimensionless → stamped units are dimensionless string.
    assert tree.synapses.units is not None


def test_bind_mismatch_raises(tree, synapse_df):
    tree.set_units("nm")
    syn = Synapses(synapse_df, units="um")
    with pytest.raises(ValueError, match="incompatible"):
        tree.set_synapses(syn)


def test_bind_matching_units_stamps(tree, synapse_df):
    from neurosetta.ops.units import get_units

    tree.set_units("nm")
    syn = Synapses(synapse_df, units="nm")
    tree.set_synapses(syn)
    assert tree.synapses.units == get_units(tree)


def test_map_mismatch_raises(tree, synapse_df):
    tree.set_units("nm")
    # Bypass set_synapses stamp by binding then manually breaking units.
    tree.set_synapses(synapse_df)
    tree.synapses.units = "um"
    with pytest.raises(ValueError, match="incompatible"):
        tree.map_synapses()


def test_map_warns_when_unset(tree, synapse_df):
    tree.set_synapses(synapse_df)  # both dimensionless → already warned on bind
    # Clear units to force map-time unset warning.
    tree.synapses.units = None
    with pytest.warns(UserWarning, match="no units|dimensionless|lack spatial"):
        tree.map_synapses()


def test_convert_units_syncs_synapse_units(tree, synapse_df):
    tree.set_units("nm")
    tree.set_synapses(synapse_df)
    assert tree.synapses.units == tree.get_units()
    x0 = tree.synapses.coordinates[0, 0]
    tree.convert_units("um")
    assert tree.synapses.units == tree.get_units()
    assert tree.synapses.coordinates[0, 0] == pytest.approx(x0 / 1000.0)


def test_synapse_set_units_declare_only(synapse_df):
    syn = Synapses(synapse_df)
    syn.set_units("nm")
    assert syn.units is not None
    # Coordinates unchanged (declaration only).
    np.testing.assert_allclose(syn.coordinates, Synapses(synapse_df).coordinates)


def test_units_survive_filter_copy_nr(tree, synapse_df, tmp_path):
    tree.set_units("nm")
    tree.set_synapses(synapse_df)
    tree.map_synapses()
    assert tree.synapses.copy().units == tree.get_units()
    assert tree.synapses.pre.units == tree.get_units()

    path = tmp_path / "u.nr"
    tree.save_tree(path)
    from neurosetta.io import load

    loaded = load(path)
    assert loaded.synapses.units == tree.get_units()
