"""Tests for persistence-diagram conversion."""

from __future__ import annotations

import numpy as np
import pytest

from neurosetta.utils.topology import persistence_diagram, tmd, validate_diagram


def test_shape_and_normalize(binary_graph):
    result = tmd(binary_graph, "f")
    diag = persistence_diagram(result, normalize=True)
    assert diag.shape[1] == 2
    assert diag.shape[0] == len(result["birth"])
    assert np.all(diag[:, 0] <= diag[:, 1])


def test_normalize_false_preserves_order(binary_graph):
    result = tmd(binary_graph, "f")
    diag = persistence_diagram(result, normalize=False)
    np.testing.assert_allclose(diag[:, 0], result["birth"])
    np.testing.assert_allclose(diag[:, 1], result["death"])


def test_empty_diagram():
    empty = {
        "birth": np.array([], dtype=np.float64),
        "death": np.array([], dtype=np.float64),
    }
    diag = persistence_diagram(empty)
    assert diag.shape == (0, 2)


def test_validate_diagram():
    d = validate_diagram([[0.0, 1.0], [0.5, 2.0]])
    assert d.shape == (2, 2)
    with pytest.raises(ValueError):
        validate_diagram([[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError):
        validate_diagram([[0.0, np.inf]])
