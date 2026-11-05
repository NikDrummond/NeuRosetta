"""Sholl intersection counts.

Cumulative distance (``path_length``, ``euclidean_length``)
    An edge from ``u`` to ``v`` with accumulated distances ``d(u) <= d(v)``
    crosses radius ``r`` when

        d(u) < r <= d(v).

    The interval is half-open at the source, so a vertex that sits on a shell
    is counted on its incoming edge and not again on the outgoing edges.
    Radius 0 is not crossed by an edge that starts at distance 0. A
    zero-length edge crosses nothing. One long edge can cross many radii;
    the count is a difference array, O(E + R) after the distances exist.

Radial distance
    Intersections are roots of the segment–sphere quadratic, not a histogram
    of vertex radii. An edge with both endpoints outside the sphere can cross
    it twice. Parameter ``t`` runs from 0 at the source to 1 at the target.
    A root is kept when ``SHOLL_T_TOL < t <= 1 + SHOLL_T_TOL``. Tangencies
    (discriminant within tolerance of 0) are not transverse crossings and are
    not counted. A degenerate zero-length segment is not a crossing.
    Radius 0 is reported as 0: a point is not a shell crossing.

Branch points that lie on a shell are counted once, via the incoming edge.
"""

from __future__ import annotations

from numpy import (
    add,
    asarray,
    count_nonzero,
    cumsum,
    divide,
    einsum,
    empty,
    float64,
    full_like,
    int64,
    isfinite,
    maximum,
    ndarray,
    searchsorted,
    sqrt,
    zeros,
)

# Absolute tolerance on the segment parameter t (dimensionless, in [0, 1]).
SHOLL_T_TOL = 1e-8


def validate_radii(radii) -> ndarray:
    """Return radii as a finite, strictly increasing, nonnegative 1-d array.

    Parameters
    ----------
    radii : array-like
        Candidate radii. Radius 0 is allowed and produces no ordinary
        intersections under the conventions above.

    Returns
    -------
    ndarray
        Copy of *radii* as ``float64``.

    Raises
    ------
    ValueError
        If *radii* is not one-dimensional, empty, non-finite, negative,
        unsorted, or contains duplicates.
    """
    arr = asarray(radii, dtype=float64)
    if arr.ndim != 1:
        raise ValueError(f"radii must be one-dimensional, got shape {arr.shape}.")
    if arr.size == 0:
        raise ValueError("radii must contain at least one value.")
    if not isfinite(arr).all():
        raise ValueError("radii must be finite.")
    if (arr < 0).any():
        raise ValueError("radii must be nonnegative.")
    if arr.size > 1 and (arr[1:] <= arr[:-1]).any():
        raise ValueError("radii must be strictly increasing, without duplicates.")
    return arr.copy()


def count_distance_crossings(distance: ndarray, edges: ndarray, radii: ndarray) -> ndarray:
    """Count half-open threshold crossings ``d_source < r <= d_target``.

    Parameters
    ----------
    distance : ndarray
        Accumulated distance at each vertex.
    edges : ndarray
        Directed edges, shape ``(E, 2)``, oriented away from the root.
    radii : ndarray
        Strictly increasing nonnegative radii.

    Returns
    -------
    ndarray
        ``int64`` intersection count per radius.

    Raises
    ------
    ValueError
        If a directed edge moves closer to the root. Weights must be
        nonnegative on an outward-oriented tree.
    """
    src = edges[:, 0].astype(int64, copy=False)
    tgt = edges[:, 1].astype(int64, copy=False)
    d0 = asarray(distance, dtype=float64)[src]
    d1 = asarray(distance, dtype=float64)[tgt]
    if (d1 + 1e-9 < d0).any():
        raise ValueError(
            "Cumulative distance decreases along a directed edge. "
            "Edge weights must be nonnegative on a tree oriented away from the root."
        )
    n_radii = int(radii.size)
    left = searchsorted(radii, d0, side="right")
    right = searchsorted(radii, d1, side="right")
    diff = zeros(n_radii + 1, dtype=int64)
    add.at(diff, left, 1)
    add.at(diff, right, -1)
    return cumsum(diff[:-1])


def count_radial_intersections(
    coords: ndarray,
    edges: ndarray,
    radii: ndarray,
    center: ndarray,
    *,
    t_tol: float = SHOLL_T_TOL,
) -> ndarray:
    """Count transverse segment–sphere intersections for every radius.

    Parameters
    ----------
    coords : ndarray
        Vertex coordinates, shape ``(V, 3)``.
    edges : ndarray
        Directed edges, shape ``(E, 2)``.
    radii : ndarray
        Nonnegative radii. Radius 0 contributes 0.
    center : ndarray
        Sphere center, shape ``(3,)``.
    t_tol : float, optional
        Absolute tolerance on the segment parameter. Roots with
        ``t <= t_tol`` belong to the source vertex and are not counted on
        this edge.

    Returns
    -------
    ndarray
        ``int64`` intersection count per radius.
    """
    if edges.size == 0:
        return zeros(radii.size, dtype=int64)
    src = edges[:, 0].astype(int64, copy=False)
    tgt = edges[:, 1].astype(int64, copy=False)
    start = coords[src]
    stop = coords[tgt]
    direction = stop - start
    offset = start - asarray(center, dtype=float64).reshape(1, 3)
    a = einsum("ij,ij->i", direction, direction)
    b = 2.0 * einsum("ij,ij->i", offset, direction)
    ff = einsum("ij,ij->i", offset, offset)
    scale = maximum(1.0, maximum(sqrt(a), sqrt(ff)))
    degenerate = a <= (t_tol * scale) ** 2
    counts = zeros(radii.size, dtype=int64)
    for i, radius in enumerate(asarray(radii, dtype=float64)):
        if radius == 0.0:
            continue
        c = ff - float(radius) * float(radius)
        disc = b * b - 4.0 * a * c
        disc_scale = maximum(scale, float(radius))
        disc_tol = (t_tol * disc_scale * disc_scale) ** 2
        transverse = (~degenerate) & (disc > disc_tol)
        sqrt_disc = sqrt(maximum(disc, 0.0))
        sign_b = asarray(b >= 0.0, dtype=float64) * 2.0 - 1.0
        q = -0.5 * (b + sign_b * sqrt_disc)
        t1 = divide(q, a, out=full_like(q, float("nan")), where=transverse)
        q_ok = transverse & (q != 0.0)
        t2 = divide(c, q, out=full_like(q, float("nan")), where=q_ok)
        hit1 = transverse & (t1 > t_tol) & (t1 <= 1.0 + t_tol)
        hit2 = q_ok & (t2 > t_tol) & (t2 <= 1.0 + t_tol)
        counts[i] = int(count_nonzero(hit1) + count_nonzero(hit2))
    return counts


def vertex_coordinates(g) -> ndarray:
    """Return ``(V, 3)`` coordinates from vertex properties ``x``, ``y``, ``z``."""
    missing = [name for name in ("x", "y", "z") if name not in g.vp]
    if missing:
        raise ValueError(
            f"Radial Sholl requires vertex properties 'x', 'y', and 'z'; missing {missing}."
        )
    return asarray(
        [g.vp["x"].a, g.vp["y"].a, g.vp["z"].a],
        dtype=float64,
    ).T.copy()


def edge_list(g) -> ndarray:
    """Return directed edges as an ``(E, 2)`` integer array."""
    edges = asarray(g.get_edges(), dtype=int64)
    if edges.size == 0:
        return empty((0, 2), dtype=int64)
    return edges
