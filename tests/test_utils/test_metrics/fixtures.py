"""Deterministic 3D morphology fixtures for metric invariance tests."""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import array, ones

from neurosetta.api import Tree
from neurosetta.utils.graph_utils.node_types import infer_node_types

# Fixed, asymmetric, non-planar, branched morphology for invariance contracts.
# Coordinates are intentionally irregular so PCA eigenvalues are non-degenerate.
_INVARIANCE_EDGES = array(
    [
        [0, 1],
        [1, 2],
        [1, 3],
        [3, 4],
        [3, 5],
        [5, 6],
        [5, 7],
        [7, 8],
        [7, 9],
        [2, 10],
        [10, 11],
        [4, 12],
    ],
    dtype=int,
)

_INVARIANCE_COORDS = array(
    [
        [0.0, 0.0, 0.0],  # 0 root
        [1.2, 0.3, 0.4],  # 1
        [2.5, -0.8, 1.1],  # 2
        [0.4, 2.1, 0.7],  # 3
        [-0.9, 3.0, 1.8],  # 4
        [1.8, 2.7, -0.5],  # 5
        [3.2, 3.5, -1.2],  # 6
        [2.1, 4.4, 0.9],  # 7
        [3.6, 5.1, 1.7],  # 8
        [1.0, 5.3, 0.2],  # 9
        [3.4, -1.5, 2.0],  # 10
        [4.7, -0.6, 2.8],  # 11
        [-2.0, 3.8, 2.5],  # 12
    ],
    dtype=float,
)

# Completely different nondegenerate coordinates (same topology) for
# topology-domain coordinate-independence tests.
_ALT_COORDS = array(
    [
        [10.0, -4.0, 2.0],
        [11.5, -3.2, 3.1],
        [13.0, -5.0, 4.4],
        [9.2, -1.0, 1.5],
        [7.5, 0.5, 0.2],
        [10.8, 0.2, -1.0],
        [12.4, 1.5, -2.2],
        [11.1, 2.8, 0.6],
        [13.2, 3.9, 1.8],
        [9.6, 3.5, -0.4],
        [14.5, -6.1, 5.0],
        [16.0, -4.8, 6.2],
        [6.0, 1.8, -0.8],
    ],
    dtype=float,
)


def _tree_from_edges_coords(
    edges,
    coords,
    *,
    tree_id: int = 1,
    units: str = "micron",
) -> Tree:
    g = Graph(directed=True)
    g.add_vertex(len(coords))
    g.add_edge_list(edges)
    g.vp["x"] = g.new_vp("double", coords[:, 0])
    g.vp["y"] = g.new_vp("double", coords[:, 1])
    g.vp["z"] = g.new_vp("double", coords[:, 2])
    g.vp["radius"] = g.new_vp("double", ones(len(coords)) * 0.5)
    g.vp["node_type"] = g.new_vp("int", infer_node_types(g))
    return Tree(
        ID=tree_id,
        metadata={"isReduced": False, "synthetic": True, "units": units},
        graph=g,
    )


def make_invariance_tree(*, units: str = "micron") -> Tree:
    """Return a fixed asymmetric 3D branched tree for invariance tests."""
    return _tree_from_edges_coords(_INVARIANCE_EDGES, _INVARIANCE_COORDS, units=units)


def make_invariance_tree_alt_coords(*, units: str = "micron") -> Tree:
    """Same topology as :func:`make_invariance_tree` with different coordinates."""
    return _tree_from_edges_coords(_INVARIANCE_EDGES, _ALT_COORDS, units=units)
