"""Headless morphology summary on the analytical reference trees.

Writes ``examples/output/morphology_summary.csv``. No GUI.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from neurosetta.api import Forest
from neurosetta.testing import (
    asymmetric_branch,
    irregular_curve,
    straight_chain,
    symmetric_binary,
)

OUT = Path(__file__).resolve().parent / "output"


def main() -> None:
    trees = [
        straight_chain(),
        symmetric_binary(),
        asymmetric_branch(),
        irregular_curve(),
    ]
    forest = Forest(trees)
    rows = []
    for tree in forest:
        rows.append(
            {
                "id": tree.ID,
                "nodes": tree.count_nodes(),
                "edges": tree.count_edges(),
                "branches": tree.count_branches(),
                "leaves": tree.count_leaves(),
                "cable": tree.get_total_cable_length(),
                "max_strahler": tree.get_max_strahler_order(),
            }
        )
    frame = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "morphology_summary.csv"
    frame.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(frame["id"].astype(str), frame["cable"])
    ax.set_xlabel("tree id")
    ax.set_ylabel("cable")
    fig.tight_layout()
    fig.savefig(OUT / "cable.png", dpi=120)
    plt.close(fig)
    print(frame.to_string(index=False))
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
