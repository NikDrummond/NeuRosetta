"""TMD graph-property persistence through ``.nr``."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from neurosetta.io import load, save
from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.topology import TMD_GP


def test_tmd_roundtrips_in_nr():
    tree = make_synthetic_tree(n=40, seed=11)
    tree.get_root_distance(bind=True)
    tree.tmd(func="Root_distance", bind=True)
    assert TMD_GP in tree.graph.gp
    original = tree.graph.gp[TMD_GP]

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir)
        save(tree, path)
        loaded = load(path / f"{tree.name}.nr")

    assert TMD_GP in loaded.graph.gp
    restored = loaded.graph.gp[TMD_GP]
    assert restored["function"] == "Root_distance"
    np.testing.assert_array_equal(restored["birth_ind"], original["birth_ind"])
    np.testing.assert_array_equal(restored["death_ind"], original["death_ind"])
    np.testing.assert_allclose(restored["survival_len"], original["survival_len"])

    # Cached TMD is reused after load
    again = loaded.tmd(func="Root_distance", bind=False, recalculate=False)
    assert again is not None
    np.testing.assert_allclose(again["survival_len"], original["survival_len"])
