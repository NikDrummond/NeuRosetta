"""Tests for connectivity-table → Synapses extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neurosetta import extract_synapses
from neurosetta.core.synapses import Synapses


@pytest.fixture
def connectivity_df() -> pd.DataFrame:
    # synapse 1: 10 → 20
    # synapse 2: 20 → 10
    # synapse 3: 10 → 30
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "pre": [10, 20, 10],
            "post": [20, 10, 30],
            "pre_x": [0.0, 1.0, 2.0],
            "pre_y": [0.0, 1.0, 2.0],
            "pre_z": [0.0, 1.0, 2.0],
            "post_x": [3.0, 4.0, 5.0],
            "post_y": [3.0, 4.0, 4.0],
            "post_z": [3.0, 4.0, 5.0],
            "cleft": [1.0, 2.0, 3.0],
        }
    )


def test_extract_single_neuron(connectivity_df):
    syn = extract_synapses(connectivity_df, 10)
    assert isinstance(syn, Synapses)
    assert syn.owner_id == 10
    assert len(syn) == 3

    df = syn.to_dataframe()
    # post rows first (synapse 2: 20→10), then pre (synapses 1, 3)
    assert list(df["type"]) == ["post", "pre", "pre"]
    assert list(df["synapse_id"]) == [2, 1, 3]
    assert list(df["partner_id"]) == [20, 20, 30]
    np.testing.assert_allclose(df.loc[0, ["x", "y", "z"]], [4.0, 4.0, 4.0])
    np.testing.assert_allclose(df.loc[1, ["x", "y", "z"]], [0.0, 0.0, 0.0])


def test_extract_batch_and_extras(connectivity_df):
    tables = extract_synapses(connectivity_df, [10, 20, 999], include_extra=True)
    assert isinstance(tables, list)
    assert [t.owner_id for t in tables] == [10, 20, 999]
    assert [len(t) for t in tables] == [3, 2, 0]
    assert "cleft" in tables[0].to_dataframe().columns
    assert "cleft" not in extract_synapses(connectivity_df, 10).to_dataframe().columns


def test_extract_string_id_is_scalar():
    df = pd.DataFrame(
        {
            "id": [1],
            "pre": ["A"],
            "post": ["B"],
            "pre_x": [0.0],
            "pre_y": [0.0],
            "pre_z": [0.0],
            "post_x": [1.0],
            "post_y": [1.0],
            "post_z": [1.0],
        }
    )
    syn = extract_synapses(df, "A")
    assert isinstance(syn, Synapses)
    assert syn.owner_id == "A"
    assert len(syn) == 1
    assert syn.to_dataframe().iloc[0]["type"] == "pre"


def test_extract_missing_columns_and_bad_xyz(connectivity_df):
    with pytest.raises(KeyError, match="missing"):
        extract_synapses(connectivity_df.drop(columns=["pre"]), 10)
    with pytest.raises(ValueError, match="3 names"):
        extract_synapses(
            connectivity_df,
            10,
            pre_xyz_cols=("pre_x", "pre_y"),
        )


def test_extract_batch_sort_path(connectivity_df):
    # ≥ _SORT_THRESHOLD ids exercises the sorted lookup branch
    tables = extract_synapses(connectivity_df, [10, 20, 30])
    assert [len(t) for t in tables] == [3, 2, 1]


def test_extract_set_units(connectivity_df):
    syn = extract_synapses(connectivity_df, 10, set_units="nm")
    assert syn.units is not None
    tables = extract_synapses(connectivity_df, [10, 20], set_units="micron")
    assert all(t.units is not None for t in tables)


def test_extract_voxel_units_require_spec(connectivity_df):
    with pytest.raises(ValueError, match="voxel"):
        extract_synapses(connectivity_df, 10, set_units="voxel")
    with pytest.raises(ValueError, match="require set_units"):
        extract_synapses(connectivity_df, 10, voxel_size=4.0, voxel_unit="nm")
    syn = extract_synapses(connectivity_df, 10, set_units="voxel", voxel_size=4.0, voxel_unit="nm")
    assert syn.units is not None


def test_import_synapses_set_units():
    from neurosetta import import_synapses

    df = pd.DataFrame(
        {
            "synapse_id": [1],
            "type": ["pre"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "partner_id": [2],
        }
    )
    syn = import_synapses(df, set_units="nm")
    assert syn.units is not None
