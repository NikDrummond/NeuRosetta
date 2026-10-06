"""DFS visitors for topological morphology descriptors."""

from __future__ import annotations

import numpy as np
from graph_tool.all import DFSVisitor, VertexPropertyMap
from numpy.typing import NDArray


class TMDVisitor(DFSVisitor):
    """Bottom-up Topological Morphology Descriptor (Kanari et al., 2018).

    Runs on the **original** directed tree (parent → child). Persistence pairs
    retain original graph-tool vertex indices.

    Semantics
    ---------
    - Each leaf starts an active component identified by that leaf.
    - Degree-1 continuation vertices propagate the surviving descendant leaf.
    - At a branch, child components merge by the elder rule: the descendant leaf
      with the largest filtration value survives toward the root; every loser
      produces a persistence pair whose birth index is the branch vertex and
      whose death index is the losing leaf.
    - At the root, the final surviving component is paired with the root.

    Notes
    -----
    For a monotonic root-distance filtration, the branch/root value is normally
    the lower endpoint and the leaf value the upper endpoint. NeuRosetta keeps
    the historical naming: ``birth_ind`` = branch/root, ``death_ind`` = leaf.
    """

    def __init__(
        self,
        func: VertexPropertyMap,
        n_vertices: int,
        root: int,
    ) -> None:
        self.values: NDArray[np.floating] = func.a
        self.root = int(root)
        self.parent: NDArray[np.int64] = np.full(n_vertices, -1, dtype=np.int64)
        self.survivor: NDArray[np.int64] = np.full(n_vertices, -1, dtype=np.int64)
        self.birth_ind: list[int] = []
        self.death_ind: list[int] = []

    def tree_edge(self, e) -> None:
        source = int(e.source())
        target = int(e.target())
        self.parent[target] = source

    def finish_vertex(self, v) -> None:
        v = int(v)

        # No child has propagated a survivor → v is a leaf.
        if self.survivor[v] == -1:
            self.survivor[v] = v

        # Root claims the final remaining component.
        if v == self.root:
            self.birth_ind.append(v)
            self.death_ind.append(int(self.survivor[v]))
            return

        p = int(self.parent[v])
        candidate = int(self.survivor[v])

        # First child component reaching the parent.
        if self.survivor[p] == -1:
            self.survivor[p] = candidate
            return

        incumbent = int(self.survivor[p])

        # Elder rule: larger filtration value survives.
        if self.values[candidate] > self.values[incumbent]:
            self.birth_ind.append(p)
            self.death_ind.append(incumbent)
            self.survivor[p] = candidate
        else:
            self.birth_ind.append(p)
            self.death_ind.append(candidate)
