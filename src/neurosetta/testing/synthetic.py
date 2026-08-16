"""Procedurally generated trees for local development and tests."""

from __future__ import annotations

import math
import random
from typing import Literal

from graph_tool.all import Graph

from ..api import Forest, Tree
from ..utils.graph_utils.node_types import infer_node_types

_CHILD_WEIGHTS = (0.20, 0.45, 0.25, 0.10)
_CHILD_COUNTS = (1, 2, 3, 4)


def _rng(seed: int | None) -> random.Random:
    return random.Random(seed)


def _synthetic_tree_graph(
    rng: random.Random,
    n: int,
    *,
    level_spacing: float = 1.0,
    width: float = 1.5,
    z_variation: float = 1.0,
    radius: float = 0.5,
    root_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Graph:
    """Build a directed synthetic morphology graph with x/y/z coordinates."""
    if n < 1:
        raise ValueError("n must be at least 1")

    g = Graph(directed=True)
    g.vp["x"] = g.new_vertex_property("double")
    g.vp["y"] = g.new_vertex_property("double")
    g.vp["z"] = g.new_vertex_property("double")
    g.vp["radius"] = g.new_vertex_property("double")

    rx, ry, rz = root_xyz
    root = g.add_vertex()
    g.vp["x"][root] = rx
    g.vp["y"][root] = ry
    g.vp["z"][root] = rz
    g.vp["radius"][root] = radius

    frontier: list[tuple] = [(root, rx, ry, rz)]

    while g.num_vertices() < n and frontier:
        parent, px, py, pz = frontier.pop(0)
        remaining = n - g.num_vertices()
        if remaining <= 0:
            break

        if remaining == 1:
            num_children = 1
        else:
            num_children = rng.choices(_CHILD_COUNTS, weights=_CHILD_WEIGHTS, k=1)[0]
            num_children = min(num_children, remaining)

        angle_offset = rng.uniform(0, 2 * math.pi)

        for i in range(num_children):
            angle = angle_offset + 2 * math.pi * i / num_children + rng.uniform(-0.4, 0.4)
            rad = width * (0.7 + rng.random() * 0.6)
            x = px + rad * math.cos(angle)
            y = py + rad * math.sin(angle)
            z = pz + level_spacing + rng.uniform(-z_variation, z_variation)

            child = g.add_vertex()
            g.add_edge(parent, child)
            g.vp["x"][child] = x
            g.vp["y"][child] = y
            g.vp["z"][child] = z
            g.vp["radius"][child] = radius * (0.8 + 0.4 * rng.random())

            frontier.append((child, x, y, z))

            if g.num_vertices() >= n:
                break

    g.vp["node_type"] = g.new_vertex_property("int", infer_node_types(g))
    return g


def _forest_offsets(
    n_trees: int,
    *,
    spacing: float,
    layout: Literal["line", "grid", "random"],
    rng: random.Random,
) -> list[tuple[float, float, float]]:
    if layout == "line":
        return [(i * spacing, 0.0, 0.0) for i in range(n_trees)]

    if layout == "grid":
        cols = max(1, math.ceil(math.sqrt(n_trees)))
        return [((i % cols) * spacing, (i // cols) * spacing, 0.0) for i in range(n_trees)]

    return [
        (rng.uniform(-spacing, spacing), rng.uniform(-spacing, spacing), 0.0)
        for _ in range(n_trees)
    ]


def _synthetic_metadata(*, units: str | None) -> dict:
    meta = {"isReduced": False, "synthetic": True}
    if units is not None:
        meta["units"] = units
    return meta


def make_synthetic_tree(
    n: int = 50,
    *,
    tree_id: int = 1,
    seed: int | None = 42,
    level_spacing: float = 1.0,
    width: float = 1.5,
    z_variation: float = 1.0,
    radius: float = 0.5,
    root_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    units: str | None = None,
) -> Tree:
    """Return a procedurally generated :class:`~neurosetta.api.Tree`."""
    graph = _synthetic_tree_graph(
        _rng(seed),
        n,
        level_spacing=level_spacing,
        width=width,
        z_variation=z_variation,
        radius=radius,
        root_xyz=root_xyz,
    )
    return Tree(
        ID=tree_id,
        metadata=_synthetic_metadata(units=units),
        graph=graph,
    )


def make_synthetic_forest(
    n_trees: int = 3,
    *,
    n: int = 50,
    seed: int | None = 42,
    tree_ids: list[int] | None = None,
    tree_spacing: float = 10.0,
    layout: Literal["line", "grid", "random"] = "line",
    level_spacing: float = 1.0,
    width: float = 1.5,
    z_variation: float = 1.0,
    radius: float = 0.5,
    units: str | None = None,
) -> Forest:
    """Return a small :class:`~neurosetta.api.Forest` of synthetic trees."""
    if n_trees < 1:
        raise ValueError("n_trees must be at least 1")

    rng = _rng(seed)
    ids = list(range(1, n_trees + 1)) if tree_ids is None else list(tree_ids)
    if len(ids) != n_trees:
        raise ValueError("tree_ids length must match n_trees")

    offsets = _forest_offsets(n_trees, spacing=tree_spacing, layout=layout, rng=rng)
    trees = [
        make_synthetic_tree(
            n,
            tree_id=tree_id,
            seed=None if seed is None else seed + i,
            level_spacing=level_spacing,
            width=width,
            z_variation=z_variation,
            radius=radius,
            root_xyz=offset,
            units=units,
        )
        for i, (tree_id, offset) in enumerate(zip(ids, offsets, strict=True))
    ]
    return Forest(trees)
