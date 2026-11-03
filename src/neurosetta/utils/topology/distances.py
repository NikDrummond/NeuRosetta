"""Persistence-diagram distances (GUDHI + Kanari barcode density)."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from ._gudhi import require_gudhi
from .diagrams import validate_diagram


def barcode_distance(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
) -> float:
    """Kanari et al. barcode density distance (exact event sweep).

    For each barcode, define density at ``x`` as the number of intervals alive
    at ``x``. The distance is

        ∫ |density_A(x) - density_B(x)| dx

    This is **not** bottleneck or Wasserstein distance. Implementation is an
    ``O((N + M) log(N + M))`` sweep over interval endpoints (no rasterization).

    Intervals are treated as half-open ``[lo, hi)`` in the sense that a
    zero-length interval contributes nothing and density changes at endpoints
    are applied after accumulating area on each open segment between event
    positions.
    """
    a = validate_diagram(diagram_a)
    b = validate_diagram(diagram_b)

    if a.shape[0] == 0 and b.shape[0] == 0:
        return 0.0

    # Normalise endpoints so lo <= hi.
    if a.shape[0]:
        a_lo = np.minimum(a[:, 0], a[:, 1])
        a_hi = np.maximum(a[:, 0], a[:, 1])
    else:
        a_lo = a_hi = np.empty(0, dtype=np.float64)

    if b.shape[0]:
        b_lo = np.minimum(b[:, 0], b[:, 1])
        b_hi = np.maximum(b[:, 0], b[:, 1])
    else:
        b_lo = b_hi = np.empty(0, dtype=np.float64)

    # Events: (position, delta_density_diff)
    # A: +1 at lo, -1 at hi; B: -1 at lo, +1 at hi  →  d = dens_A - dens_B
    n_events = 2 * (a_lo.size + b_lo.size)
    if n_events == 0:
        return 0.0

    positions = np.empty(n_events, dtype=np.float64)
    deltas = np.empty(n_events, dtype=np.int64)
    i = 0
    for lo, hi in zip(a_lo, a_hi, strict=True):
        positions[i] = lo
        deltas[i] = 1
        i += 1
        positions[i] = hi
        deltas[i] = -1
        i += 1
    for lo, hi in zip(b_lo, b_hi, strict=True):
        positions[i] = lo
        deltas[i] = -1
        i += 1
        positions[i] = hi
        deltas[i] = 1
        i += 1

    order = np.argsort(positions, kind="mergesort")
    positions = positions[order]
    deltas = deltas[order]

    area = 0.0
    d = 0
    prev = positions[0]
    idx = 0
    n = positions.size
    while idx < n:
        x = positions[idx]
        if x > prev:
            area += abs(d) * (x - prev)
            prev = x
        # Apply all events at this x before moving on.
        while idx < n and positions[idx] == x:
            d += int(deltas[idx])
            idx += 1

    return float(area)


def bottleneck_distance(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
    **kwargs,
) -> float:
    """Bottleneck distance via GUDHI."""
    gudhi = require_gudhi()
    a = validate_diagram(diagram_a)
    b = validate_diagram(diagram_b)
    return float(gudhi.bottleneck_distance(a, b, **kwargs))


def wasserstein_distance(
    diagram_a: np.ndarray,
    diagram_b: np.ndarray,
    order: float = 1.0,
    internal_p: float = np.inf,
    **kwargs,
) -> float:
    """Wasserstein distance via GUDHI Hera (no POT dependency)."""
    require_gudhi()
    from gudhi.hera import wasserstein_distance as _gudhi_wasserstein_distance

    a = validate_diagram(diagram_a)
    b = validate_diagram(diagram_b)
    return float(_gudhi_wasserstein_distance(a, b, order=order, internal_p=internal_p, **kwargs))


def _resolve_n_jobs(n_jobs: int | None) -> int:
    """Map ``n_jobs`` to a concrete positive worker count."""
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return int(n_jobs)


def _barcode_pair_task(payload: tuple[int, int, np.ndarray, np.ndarray]) -> tuple[int, int, float]:
    """Picklable worker for process-pool barcode pairwise distances."""
    i, j, a, b = payload
    return i, j, barcode_distance(a, b)


def _pairwise_upper_triangle(
    diagrams: Sequence[np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_jobs: int | None = 1,
    show_progress: bool = False,
    use_processes: bool = False,
    desc: str = "Pairwise distances",
) -> NDArray[np.float64]:
    """Upper-triangle pairwise distances mirrored to a full matrix."""
    n = len(diagrams)
    out = np.zeros((n, n), dtype=np.float64)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return out

    workers = _resolve_n_jobs(n_jobs)

    if workers == 1:
        it: Sequence[tuple[int, int]] = (
            tqdm(pairs, total=len(pairs), desc=desc) if show_progress else pairs
        )
        for i, j in it:
            d = float(metric_fn(diagrams[i], diagrams[j]))
            out[i, j] = d
            out[j, i] = d
        return out

    if use_processes and metric_fn is barcode_distance:
        payloads = [(i, j, diagrams[i], diagrams[j]) for i, j in pairs]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_barcode_pair_task, p) for p in payloads]
            done = as_completed(futures)
            if show_progress:
                done = tqdm(done, total=len(futures), desc=desc)
            for fut in done:
                i, j, d = fut.result()
                out[i, j] = d
                out[j, i] = d
        return out

    # Threads: works with closures / non-picklable callables; GUDHI-friendly.
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(metric_fn, diagrams[i], diagrams[j]): (i, j) for i, j in pairs}
        done = as_completed(futures)
        if show_progress:
            done = tqdm(done, total=len(futures), desc=desc)
        for fut in done:
            i, j = futures[fut]
            d = float(fut.result())
            out[i, j] = d
            out[j, i] = d
    return out


def pairwise_distances(
    diagrams: Sequence[np.ndarray],
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "wasserstein",
    n_jobs: int | None = None,
    *,
    show_progress: bool = False,
    **kwargs,
) -> NDArray[np.float64]:
    """Pairwise persistence-diagram distance matrix (``N x N``).

    Supported metric names: ``\"bottleneck\"``, ``\"wasserstein\"``,
    ``\"sliced_wasserstein\"``, ``\"barcode\"``. Callables are also accepted.

    Parameters
    ----------
    diagrams : sequence of ndarray
        Persistence diagrams of shape ``(N_i, 2)``.
    metric : str or callable, optional
        Distance name or ``(diag_a, diag_b) -> float``. By default ``\"wasserstein\"``.
    n_jobs : int or None, optional
        Worker count. ``None`` / ``0`` / ``1`` → serial; ``-1`` → all CPUs.
        Forwarded to GUDHI for GUDHI-backed metrics. By default ``None`` (serial).
    show_progress : bool, optional
        Show a tqdm progress bar over upper-triangle pairs. By default False.
    **kwargs
        Forwarded to the distance implementation / GUDHI pairwise helper.
    """
    diags = [validate_diagram(d) for d in diagrams]
    n = len(diags)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    workers = _resolve_n_jobs(n_jobs)

    if callable(metric):
        return _pairwise_upper_triangle(
            diags,
            metric,
            n_jobs=workers,
            show_progress=show_progress,
            use_processes=False,
        )

    name = str(metric).lower()
    if name == "barcode":
        return _pairwise_upper_triangle(
            diags,
            barcode_distance,
            n_jobs=workers,
            show_progress=show_progress,
            use_processes=True,
        )

    require_gudhi()
    from gudhi.representations.metrics import (
        pairwise_persistence_diagram_distances as _gudhi_pairwise,
    )

    # GUDHI sliced Wasserstein requires num_directions.
    if name == "sliced_wasserstein" and "num_directions" not in kwargs:
        kwargs = {**kwargs, "num_directions": 10}

    # When progress is requested, use our threaded wrapper (GUDHI has no hook).
    if show_progress and n > 1:
        if name == "bottleneck":
            return _pairwise_upper_triangle(
                diags,
                bottleneck_distance,
                n_jobs=workers,
                show_progress=True,
                use_processes=False,
            )
        if name == "wasserstein":

            def _w(a: np.ndarray, b: np.ndarray) -> float:
                return wasserstein_distance(a, b, **kwargs)

            return _pairwise_upper_triangle(
                diags,
                _w,
                n_jobs=workers,
                show_progress=True,
                use_processes=False,
            )

    # GUDHI uses joblib threads; n_jobs=None means serial there — pass concrete count.
    gudhi_jobs = None if workers == 1 else workers
    return np.asarray(
        _gudhi_pairwise(diags, metric=name, n_jobs=gudhi_jobs, **kwargs),
        dtype=np.float64,
    )
