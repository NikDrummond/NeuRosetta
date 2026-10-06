"""SoA Synapses backend tests."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neurosetta.api import Tree
from neurosetta.core.synapses import (
    PAYLOAD_FORMAT,
    PAYLOAD_VERSION,
    TYPE_POST,
    TYPE_PRE,
    SynapseRow,
    Synapses,
    synapses_from_arrays,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*units.*:UserWarning")


@pytest.fixture
def tree(test_tree) -> Tree:
    t = Tree(ID=test_tree.ID, metadata=dict(test_tree.metadata), graph=test_tree.graph)
    t.set_units("nm")
    return t


def test_construct_from_df_custom_col_and_filter():
    df = pd.DataFrame(
        {
            "synapse_id": [1, 2, 3],
            "type": ["pre", "post", "output"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "partner_id": [10, 20, 10],
            "conf": [0.9, 0.5, 0.1],
        }
    )
    s = Synapses(df)
    assert "conf" in s.columns
    assert "conf" in s.annotations
    assert s.to_dataframe()["conf"].iloc[0] == 0.9
    assert len(s.filter(type="pre")) == 2
    assert len(s.filter(conf=0.9)) == 1
    assert np.array_equal(s.type_codes[:1], [TYPE_PRE])
    assert int(s.type_codes[1]) == int(TYPE_POST)


def test_pickle_v2_roundtrip():
    s = Synapses(
        {
            "synapse_id": [1],
            "type": ["pre"],
            "x": [0.0],
            "y": [0.0],
            "z": [0.0],
            "partner_id": [2],
            "conf": [0.9],
        }
    )
    payload = s._to_payload()
    assert payload["format"] == PAYLOAD_FORMAT
    assert payload["version"] == PAYLOAD_VERSION
    assert "df" not in payload

    s2 = pickle.loads(pickle.dumps(s))
    assert len(s2) == 1
    assert s2.annotations["conf"][0] == 0.9
    assert s2.index_of_id(1) == 0


def test_legacy_v1_state_load():
    df = pd.DataFrame(
        {
            "synapse_id": [7, 8],
            "type": ["pre", "post"],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "partner_id": [1, 2],
        }
    )
    state = {"df": df, "mapping_meta": {"valid": False}}
    bare = Synapses.__new__(Synapses)
    bare.__setstate__(state)
    assert bare.owner_id is None
    assert len(bare) == 2
    assert list(bare.types) == ["pre", "post"]


def test_from_arrays_and_synapses_from_arrays():
    s = Synapses.from_arrays(
        type=["pre", "post"],
        coordinates=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        partner_id=[10, 20],
        score=[0.1, 0.2],
    )
    assert len(s) == 2
    assert "score" in s.annotations
    s2 = synapses_from_arrays(
        type=["pre"],
        x=[0.0],
        y=[0.0],
        z=[0.0],
        synapse_id=[99],
    )
    assert s2.synapse_ids[0] == 99


def test_concat_and_indexes():
    a = Synapses.from_arrays(
        synapse_id=[1],
        type=["pre"],
        x=[0.0],
        y=[0.0],
        z=[0.0],
        partner_id=[10],
    )
    b = Synapses.from_arrays(
        synapse_id=[2],
        type=["post"],
        x=[1.0],
        y=[0.0],
        z=[0.0],
        partner_id=[10],
    )
    c = Synapses.concat(a, b)
    assert len(c) == 2
    assert not c.is_mapped
    assert c.index_of_id(2) == 1
    assert list(c.indices_for_partner(10)) == [0, 1]
    with pytest.raises(ValueError, match="Duplicate"):
        Synapses.concat(a, a)


def test_take_mask_and_mapping_accessors():
    s = Synapses.from_arrays(
        synapse_id=[1, 2, 3],
        type=["pre", "post", "pre"],
        coordinates=np.zeros((3, 3)),
        partner_id=[1, 2, 1],
    )
    s.set_mapping(
        edge_index=np.array([0, 1, 0]),
        edge_fraction=np.array([0.1, 0.2, 0.3]),
        nearest=np.zeros((3, 3)),
        distance_to_tree=np.array([0.0, 1.0, 2.0]),
        mapped=np.array([True, True, False]),
        distance_along_edge=np.array([0.1, 0.2, 0.3]),
    )
    assert s.is_mapped
    assert list(s.indices_on_edge(0)) == [0, 2]
    sub = s.mask(s.mapped_flags)
    assert len(sub) == 2
    remapped = s.remap_edge_indices({0: 0, 1: 1}, mapped_only=True)
    assert len(remapped) == 2
    assert list(remapped.edge_index) == [0, 1]


def test_freeze_save_load_roundtrip(tree, tmp_path: Path):
    from neurosetta.io.nr_utils import load, save

    starts, ends = tree.get_edge_coordinates()
    mid = 0.5 * (starts[0] + ends[0])
    df = pd.DataFrame(
        {
            "synapse_id": [1, 2],
            "type": ["pre", "post"],
            "x": [mid[0], mid[0]],
            "y": [mid[1], mid[1]],
            "z": [mid[2], mid[2]],
            "partner_id": [100, 200],
            "conf": [0.9, 0.8],
        }
    )
    tree.set_synapses(df)
    tree.map_synapses()
    out = tmp_path / f"{tree.ID}.nr"
    save(tree, out)
    # Live object rehydrated after save
    assert tree.synapses is not None
    assert not isinstance(tree.graph.gp["synapses"], dict)

    loaded = load(out)
    syn = loaded.synapses
    assert syn is not None
    assert len(syn) == 2
    assert syn.is_mapped
    assert "conf" in syn.annotations
    assert syn.owner_id == loaded.ID


def _sample_synapses() -> Synapses:
    return Synapses(
        {
            "synapse_id": [1, 2, 3],
            "type": ["pre", "post", "pre"],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "partner_id": [10, 20, 10],
            "conf": [0.9, 0.5, 0.1],
        }
    )


def test_filter_keyword_still_works():
    s = _sample_synapses()
    assert len(s.filter(type="pre")) == 2
    assert len(s.filter(conf=0.9)) == 1
    assert list(s.filter(partner_id=[10, 20]).synapse_ids) == [1, 2, 3]


def test_filter_predicate_named_function():
    s = _sample_synapses()

    def is_pre(row):
        return row.type == "pre"

    filtered = s.filter(is_pre)
    assert list(filtered.synapse_ids) == [1, 3]


def test_filter_predicate_compound():
    s = _sample_synapses()

    def high_conf_pre(row):
        return row.type == "pre" and row["conf"] > 0.5

    assert list(s.filter(high_conf_pre).synapse_ids) == [1]


def test_filter_predicate_default_args():
    s = _sample_synapses()

    def above(row, thresh=0.4):
        return float(row["conf"]) >= thresh

    assert list(s.filter(above).synapse_ids) == [1, 2]


def test_filter_predicate_and_keywords_raises():
    s = _sample_synapses()

    def always_true(row):
        return True

    with pytest.raises(ValueError, match="not both"):
        s.filter(always_true, type="pre")


def test_filter_predicate_non_bool_raises():
    s = _sample_synapses()

    def not_bool(row):
        return 1

    with pytest.raises(TypeError, match="must return bool"):
        s.filter(not_bool)


def test_filter_predicate_wrong_arity_raises():
    s = _sample_synapses()

    def two_args(row, extra):
        return True

    with pytest.raises(TypeError, match="exactly one required argument"):
        s.filter(two_args)


def test_filter_no_criteria_raises():
    s = _sample_synapses()
    with pytest.raises(ValueError, match="predicate or at least one keyword"):
        s.filter()


def test_synapse_row_get_missing_column():
    s = _sample_synapses()
    row = SynapseRow(s, 0)
    assert row.get("missing") is None
    assert row.get("conf") == 0.9
    assert row.index == 0
    assert "conf" in row


def test_pre_post_properties_still_work():
    s = _sample_synapses()
    assert len(s.pre) == 2
    assert len(s.post) == 1
