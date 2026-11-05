"""Time subtree metrics and Sholl on one synthetic tree.

Run from the repo root with the ``nr`` environment active.
"""

from __future__ import annotations

import time

import numpy as np

from neurosetta.testing import make_synthetic_tree


def _time(fn, repeats: int = 3) -> float:
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(min(samples))


def main() -> None:
    tree = make_synthetic_tree(5_000, seed=0, units="micron")
    radii = np.linspace(1.0, 50.0, 40)
    jobs = {
        "leaf_count": lambda: tree.get_subtree_leaf_count(),
        "path_length": lambda: tree.get_subtree_path_length(),
        "height": lambda: tree.get_subtree_height(),
        "strahler": lambda: tree.get_strahler_order(),
        "sholl_path": lambda: tree.get_sholl_profile(radii, mode="path_length"),
        "sholl_radial": lambda: tree.get_sholl_profile(radii, mode="radial"),
    }
    print(f"{'metric':<16} {'seconds':>10}")
    for name, fn in jobs.items():
        print(f"{name:<16} {_time(fn):10.4f}")


if __name__ == "__main__":
    main()
