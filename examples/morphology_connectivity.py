"""Headless synapses, cable distance, and a directed connectivity table.

Two neurons, reciprocal synapses with distinct ``cleft_id`` values, mapped
onto the arbor, then reduced. Root cable distance must match before and
after reduction. The connectivity table must contain both directions.

Writes ``examples/output/connectivity.csv`` and a ``.nr`` round-trip.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import neurosetta as nr
from neurosetta.testing import asymmetric_branch, straight_chain

OUT = Path(__file__).resolve().parent / "output"


def _on_first_edge(tree: nr.Tree, n: int, kind: str, partner, cleft_prefix: str) -> pd.DataFrame:
    starts, ends = tree.get_edge_coordinates()
    point = 0.5 * (starts[0] + ends[0])
    return pd.DataFrame(
        {
            "synapse_id": np.arange(n),
            "type": [kind] * n,
            "x": np.full(n, point[0]),
            "y": np.full(n, point[1]),
            "z": np.full(n, point[2]),
            "partner_id": [partner] * n,
            "cleft_id": [f"{cleft_prefix}-{i}" for i in range(n)],
        }
    )


def main() -> None:
    a = straight_chain()
    b = asymmetric_branch()
    a.set_synapses(_on_first_edge(a, 2, "pre", b.ID, "a"))
    b.set_synapses(_on_first_edge(b, 1, "pre", a.ID, "b"))
    a.map_synapses(max_distance=None)
    b.map_synapses(max_distance=None)
    before = a.get_synapse_path_distance().copy()
    a.get_reduced_tree(inplace=True)
    after = a.get_synapse_path_distance()
    if not np.allclose(before, after, atol=1e-9):
        raise SystemExit(f"cable not preserved: {before} vs {after}")

    forest = nr.Forest([a, b])
    table = forest.get_connectivity_table()
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "connectivity.csv", index=False)
    nr.save(a, OUT / "reduced_with_synapses.nr")
    loaded = nr.load(OUT / "reduced_with_synapses.nr")
    if loaded.synapses is None or len(loaded.synapses) != 2:
        raise SystemExit("synapse round-trip lost rows")
    print(table.to_string(index=False))
    print("cable before/after", before, after)
    print("reloaded synapses", len(loaded.synapses))


if __name__ == "__main__":
    main()
