"""Persistence-diagram distances (GUDHI + Kanari barcode density)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

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


def _pairwise_symmetric(
    diagrams: Sequence[np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> NDArray[np.float64]:
    """Upper-triangle pairwise distances mirrored to a full matrix."""
    n = len(diagrams)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(metric_fn(diagrams[i], diagrams[j]))
            out[i, j] = d
            out[j, i] = d
    return out


def pairwise_distances(
    diagrams: Sequence[np.ndarray],
    metric: str | Callable[[np.ndarray, np.ndarray], float] = "wasserstein",
    n_jobs: int | None = None,
    **kwargs,
) -> NDArray[np.float64]:
    """Pairwise persistence-diagram distance matrix (``N x N``).

    Supported metric names: ``\"bottleneck\"``, ``\"wasserstein\"``,
    ``\"sliced_wasserstein\"``, ``\"barcode\"``. Callables are also accepted.

    GUDHI-backed metrics use
    ``gudhi.representations.metrics.pairwise_persistence_diagram_distances``.
    The Kanari ``\"barcode\"`` metric uses :func:`barcode_distance`.
    """
    diags = [validate_diagram(d) for d in diagrams]
    n = len(diags)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    if callable(metric):
        return _pairwise_symmetric(diags, metric)

    name = str(metric).lower()
    if name == "barcode":
        return _pairwise_symmetric(diags, barcode_distance)

    require_gudhi()
    from gudhi.representations.metrics import (
        pairwise_persistence_diagram_distances as _gudhi_pairwise,
    )

    # GUDHI sliced Wasserstein requires num_directions.
    if name == "sliced_wasserstein" and "num_directions" not in kwargs:
        kwargs = {**kwargs, "num_directions": 10}

    # Prefer Hera Wasserstein (default GUDHI "wasserstein"); avoid POT import.
    return np.asarray(
        _gudhi_pairwise(diags, metric=name, n_jobs=n_jobs, **kwargs),
        dtype=np.float64,
    )
