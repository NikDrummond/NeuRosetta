"""Benchmark synapse → edge mapping.

Excludes Numba compilation warm-up from reported timings.
"""

from __future__ import annotations

import time

import numpy as np

from neurosetta.testing import make_synthetic_tree
from neurosetta.utils.geometry_utils.segments import (
    project_points_to_segments,
    project_points_to_segments_bruteforce,
)


def _time(fn, *args, repeats: int = 3, **kwargs) -> float:
    # Warm-up (compile / cache)
    fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return float(min(times))


def main() -> None:
    configs = [
        (1_000, 1_000),
        (10_000, 5_000),
        (100_000, 2_000),
    ]
    print(f"{'E':>8} {'N':>8} {'bruteforce_s':>14} {'auto_s':>10}")
    for n_nodes, n_syn in configs:
        tree = make_synthetic_tree(n_nodes, seed=0)
        starts, ends = tree.get_edge_coordinates()
        e = starts.shape[0]
        rng = np.random.default_rng(0)
        # Synapses near random edge midpoints with small noise.
        idx = rng.integers(0, e, size=n_syn)
        mid = 0.5 * (starts[idx] + ends[idx])
        pts = mid + rng.normal(scale=0.05, size=mid.shape)

        t_bf = _time(project_points_to_segments_bruteforce, pts, starts, ends)
        t_auto = _time(project_points_to_segments, pts, starts, ends, method="auto")
        print(f"{e:8d} {n_syn:8d} {t_bf:14.4f} {t_auto:10.4f}")


if __name__ == "__main__":
    main()
