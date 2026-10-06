"""Fixtures for topology tests."""

from __future__ import annotations

import numpy as np
import pytest
from graph_tool.all import Graph


def _add_func(g: Graph, values: np.ndarray, name: str = "f") -> Graph:
    g.vp[name] = g.new_vp("double", values)
    return g


@pytest.fixture
def chain_graph() -> Graph:
    """Rooted chain: 0 → 1 → 2 → 3 with increasing filtration."""
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    g = Graph(edges, directed=True)
    return _add_func(g, np.array([0.0, 1.0, 2.0, 3.0]))


@pytest.fixture
def binary_graph() -> Graph:
    """
        0
       / \\
      1   2
     / \\
    3   4
    """
    edges = np.array([[0, 1], [0, 2], [1, 3], [1, 4]])
    g = Graph(edges, directed=True)
    return _add_func(g, np.array([0.0, 1.0, 1.0, 5.0, 3.0]))


@pytest.fixture
def multifurcation_graph() -> Graph:
    """Root with three children leaves."""
    edges = np.array([[0, 1], [0, 2], [0, 3]])
    g = Graph(edges, directed=True)
    return _add_func(g, np.array([0.0, 2.0, 5.0, 3.0]))


@pytest.fixture
def asymmetric_graph() -> Graph:
    """Asymmetric tree with a long chain on one side."""
    edges = np.array([[0, 1], [0, 2], [1, 3], [3, 4], [3, 5]])
    g = Graph(edges, directed=True)
    return _add_func(g, np.array([0.0, 1.0, 4.0, 2.0, 6.0, 3.0]))
