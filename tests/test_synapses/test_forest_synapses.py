"""Forest.set_synapses — ID-matched batch attach."""

from __future__ import annotations

import pandas as pd
import pytest

from neurosetta import Forest, Synapses, extract_synapses, set_synapses
from neurosetta.testing import make_synthetic_forest, make_synthetic_tree

pytestmark = pytest.mark.filterwarnings("ignore:.*units.*:UserWarning")


def _nr_schema_df(*, n: int = 2, partner: int = 99) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "synapse_id": list(range(n)),
            "type": ["pre"] * n,
            "x": [float(i) for i in range(n)],
            "y": [0.0] * n,
            "z": [0.0] * n,
            "partner_id": [partner] * n,
        }
    )


@pytest.fixture
def forest():
    return make_synthetic_forest(2, n=20, units="nm", tree_ids=[1, 2])


@pytest.fixture
def connectivity_df() -> pd.DataFrame:
    # 1 → 2, 2 → 1, 1 → 99
    return pd.DataFrame(
        {
            "id": [10, 11, 12],
            "pre": [1, 2, 1],
            "post": [2, 1, 99],
            "pre_x": [0.0, 1.0, 2.0],
            "pre_y": [0.0, 1.0, 2.0],
            "pre_z": [0.0, 1.0, 2.0],
            "post_x": [3.0, 4.0, 5.0],
            "post_y": [3.0, 4.0, 5.0],
            "post_z": [3.0, 4.0, 5.0],
        }
    )


def test_extract_then_set_roundtrip(forest, connectivity_df):
    syns = extract_synapses(connectivity_df, forest.ids())
    attached = forest.set_synapses(syns, missing="error")
    assert set(attached) == {1, 2}
    assert forest.by_id(1).synapses is not None
    assert forest.by_id(1).synapses.owner_id == 1
    assert len(forest.by_id(1).synapses) == 3  # post from 11 + pre 10,12
    assert len(forest.by_id(2).synapses) == 2


def test_package_dispatch_forest_and_tree(forest, connectivity_df):
    syns = extract_synapses(connectivity_df, forest.ids())
    attached = set_synapses(forest, syns, missing="error")
    assert len(attached) == 2

    tree = make_synthetic_tree(20, tree_id=7, units="nm")
    bound = set_synapses(tree, _nr_schema_df())
    assert isinstance(bound, Synapses)
    assert tree.synapses is bound
    assert bound.owner_id == 7


def test_mapping_and_int_str_id_coerce():
    tree = make_synthetic_tree(20, tree_id=7, units="nm")
    forest = Forest([tree])
    syn = Synapses(_nr_schema_df(), owner_id="7")
    attached = forest.set_synapses([syn], missing="error")
    assert 7 in attached
    assert forest[0].synapses is not None

    forest2 = Forest([make_synthetic_tree(20, tree_id=1, units="nm")])
    attached2 = forest2.set_synapses({1: _nr_schema_df()}, missing="error")
    assert 1 in attached2
    assert len(forest2.by_id(1).synapses) == 2


def test_missing_warn_and_error(forest):
    syn = Synapses(_nr_schema_df(), owner_id=1)
    with pytest.warns(UserWarning, match="No synapses found for tree ID 2"):
        forest.set_synapses([syn], missing="warn")
    with pytest.raises(ValueError, match="No synapses found for tree ID 2"):
        forest.set_synapses([syn], missing="error")


def test_unused_warn_and_error(forest):
    syns = [
        Synapses(_nr_schema_df(), owner_id=1),
        Synapses(_nr_schema_df(), owner_id=2),
        Synapses(_nr_schema_df(), owner_id=99),
    ]
    with pytest.warns(UserWarning, match="unused"):
        forest.set_synapses(syns, missing="error", unused="warn")
    with pytest.raises(ValueError, match="unused"):
        forest.set_synapses(syns, missing="error", unused="error")


def test_ambiguous_owner_ids(forest):
    syns = [
        Synapses(_nr_schema_df(), owner_id=1),
        Synapses(_nr_schema_df(), owner_id="1"),
    ]
    with pytest.raises(ValueError, match="Multiple synapse tables"):
        forest.set_synapses(syns, missing="ignore")


def test_unbound_owner_id_rejected(forest):
    syn = Synapses(_nr_schema_df())  # owner_id=None
    with pytest.raises(ValueError, match="owner_id"):
        forest.set_synapses([syn])


def test_raw_dataframe_rejected(forest, connectivity_df):
    with pytest.raises(TypeError, match="extract_synapses"):
        forest.set_synapses(connectivity_df)


def test_extract_and_set_with_units(forest, connectivity_df):
    from neurosetta.ops.units import get_units

    syns = extract_synapses(connectivity_df, forest.ids(), set_units="nm")
    assert all(s.units is not None for s in syns)
    forest.set_synapses(syns, missing="error")
    for t in forest:
        assert t.synapses is not None
        assert t.synapses.units == get_units(t)


def test_forest_set_synapses_set_units_kwarg(forest):
    from neurosetta.ops.units import get_units

    attached = forest.set_synapses(
        {1: _nr_schema_df(), 2: _nr_schema_df()},
        missing="error",
        set_units="nm",
    )
    assert len(attached) == 2
    for t in forest:
        assert t.synapses.units == get_units(t)
