"""Reference helpers for topology tests."""

from __future__ import annotations

import numpy as np
from graph_tool.all import Graph


def reference_tmd_pairs(g: Graph, values: np.ndarray, root: int = 0) -> list[tuple[int, int]]:
    """Recursive elder-rule reference (no graph-tool visitors)."""
    n = g.num_vertices()
    children: list[list[int]] = [[] for _ in range(n)]
    for e in g.edges():
        children[int(e.source())].append(int(e.target()))

    survivor = [-1] * n
    pairs: list[tuple[int, int]] = []

    def dfs(v: int) -> None:
        kids = children[v]
        if not kids:
            survivor[v] = v
            return
        for c in kids:
            dfs(c)
        surv = survivor[kids[0]]
        for c in kids[1:]:
            cand = survivor[c]
            if values[cand] > values[surv]:
                pairs.append((v, surv))
                surv = cand
            else:
                pairs.append((v, cand))
        survivor[v] = surv

    dfs(root)
    pairs.append((root, survivor[root]))
    return pairs


def make_random_rooted_tree(n: int, seed: int = 0) -> tuple[Graph, np.ndarray]:
    """Random rooted tree with random filtration values."""
    rng = np.random.default_rng(seed)
    edges = []
    for v in range(1, n):
        p = int(rng.integers(0, v))
        edges.append([p, v])
    if edges:
        g = Graph(np.asarray(edges, dtype=np.int64), directed=True)
    else:
        g = Graph(directed=True)
        g.add_vertex()
    values = rng.random(n)
    g.vp["f"] = g.new_vp("double", values)
    return g, values
