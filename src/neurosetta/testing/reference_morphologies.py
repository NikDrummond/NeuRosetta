"""Small morphologies whose measurements can be derived by hand.

These are for tests and examples. Expected numbers should be calculated from
the coordinates and the connectivity, not by calling the function under test.
"""

from __future__ import annotations

import numpy as np
from graph_tool.all import Graph

from ..api import Tree
from ..utils.graph_utils.node_types import infer_node_types


def morphology_tree(
    edges: list[tuple[int, int]],
    coords: list[tuple[float, float, float]],
    *,
    tree_id: int = 1,
    units: str = "micron",
) -> Tree:
    """Build a directed tree and declare spatial units.

    Edges are parent → child. ``coords[i]`` is vertex ``i``. Cable is the
    Euclidean length of each edge until a later reduction stores accumulated
    ``Path_length``.
    """
    xyz = np.asarray(coords, dtype=float)
    graph = Graph(directed=True)
    graph.add_vertex(len(xyz))
    if edges:
        graph.add_edge_list(np.asarray(edges, dtype=int))
    graph.vp["x"] = graph.new_vp("double", xyz[:, 0])
    graph.vp["y"] = graph.new_vp("double", xyz[:, 1])
    graph.vp["z"] = graph.new_vp("double", xyz[:, 2])
    graph.vp["radius"] = graph.new_vp("double", np.ones(len(xyz)))
    graph.vp["node_type"] = graph.new_vp("int", infer_node_types(graph))
    graph.gp["ID"] = graph.new_gp("long", int(tree_id))
    graph.gp["metadata"] = graph.new_gp(
        "object",
        {"isReduced": False, "Flag": False, "synthetic": True},
    )
    tree = Tree.from_graph(graph)
    tree.set_units(units)
    return tree


def single_node() -> Tree:
    """One root, which is also a leaf. Cable 0."""
    return morphology_tree([], [(0.0, 0.0, 0.0)], tree_id=1)


def straight_chain() -> Tree:
    """Four nodes on the x axis at unit spacing. Cable 3. All Strahler orders 1."""
    return morphology_tree(
        [(0, 1), (1, 2), (2, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
        tree_id=2,
    )


def nonuniform_chain() -> Tree:
    """Edge lengths 1, 2 and 4. Nodes at x = 0, 1, 3, 7. Cable 7."""
    return morphology_tree(
        [(0, 1), (1, 2), (2, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (7.0, 0.0, 0.0)],
        tree_id=3,
    )


def simple_bifurcation() -> Tree:
    """Root, one continuation node, then a right angle.

    Vertices: 0 (0,0,0) → 1 (1,0,0) → 2 (2,0,0) and 1 → 3 (1,1,0).
    The only bifurcation is vertex 1. Child–child angle is π/2.
    """
    return morphology_tree(
        [(0, 1), (1, 2), (1, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)],
        tree_id=4,
    )


def symmetric_binary() -> Tree:
    """Perfect binary tree of depth 2. Six unit edges. Root Strahler order 3."""
    return morphology_tree(
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)],
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (-2.0, 0.0, 0.0),
            (-1.0, 1.0, 0.0),
        ],
        tree_id=5,
    )


def asymmetric_branch() -> Tree:
    """Root with one leaf and one branch that itself has two leaves.

    Edges (0,1), (0,2), (2,3), (2,4), each of length 1.
    Unweighted partition asymmetry is 0.5 at the root and 0 at vertex 2.
    Root Strahler order is 2.
    """
    return morphology_tree(
        [(0, 1), (0, 2), (2, 3), (2, 4)],
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (1.0, 1.0, 0.0),
        ],
        tree_id=6,
    )


def multifurcation() -> Tree:
    """Root with three leaves. One branch, zero bifurcations (out-degree is 3)."""
    return morphology_tree(
        [(0, 1), (0, 2), (0, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        tree_id=7,
    )


def zero_length_edge() -> Tree:
    """A zero-length edge followed by a unit edge. Cable 1. Two edges."""
    return morphology_tree(
        [(0, 1), (1, 2)],
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        tree_id=8,
    )


def irregular_curve() -> Tree:
    """Polyline (0,0,0) → (1,0,0) → (1,2,0) → (1,2,0.5). Cable 3.5.

    The straight chord from the first vertex to the last is not 3.5.
    """
    return morphology_tree(
        [(0, 1), (1, 2), (2, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 2.0, 0.0), (1.0, 2.0, 0.5)],
        tree_id=9,
    )


def spatial_tetrahedron() -> Tree:
    """Four non-coplanar points. Convex hull volume is 1/6."""
    return morphology_tree(
        [(0, 1), (0, 2), (0, 3)],
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        tree_id=10,
    )
