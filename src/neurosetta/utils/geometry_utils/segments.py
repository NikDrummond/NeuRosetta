"""Point-to-line-segment projection primitives (Numba).

These kernels operate on plain NumPy arrays and must not import Tree.
"""

from __future__ import annotations

from numba import njit
from numpy import asarray, empty, float64, inf, int64

_JIT = dict(nogil=True, fastmath=True, cache=True)


@njit(**_JIT)
def _project_point_to_segment_scalar(px, py, pz, ax, ay, az, bx, by, bz):
    """Project point p onto segment a→b.

    Returns ``qx, qy, qz, t, dist`` with ``t`` clipped to ``[0, 1]``.
    """
    vx = bx - ax
    vy = by - ay
    vz = bz - az
    vv = vx * vx + vy * vy + vz * vz

    if vv == 0.0:
        dx = px - ax
        dy = py - ay
        dz = pz - az
        return ax, ay, az, 0.0, (dx * dx + dy * dy + dz * dz) ** 0.5

    wx = px - ax
    wy = py - ay
    wz = pz - az
    t = (wx * vx + wy * vy + wz * vz) / vv
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0

    qx = ax + t * vx
    qy = ay + t * vy
    qz = az + t * vz
    dx = px - qx
    dy = py - qy
    dz = pz - qz
    return qx, qy, qz, t, (dx * dx + dy * dy + dz * dz) ** 0.5


@njit(**_JIT)
def project_points_to_segments_bruteforce(points, segment_starts, segment_ends):
    """Exact nearest-segment projection for every point (O(N·E)).

    Tie-break: lowest edge index.
    """
    n = points.shape[0]
    e = segment_starts.shape[0]

    nearest_edge = empty(n, dtype=int64)
    nearest_point = empty((n, 3), dtype=float64)
    distance = empty(n, dtype=float64)
    t_out = empty(n, dtype=float64)

    for i in range(n):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        best_d = inf
        best_e = 0
        best_qx = 0.0
        best_qy = 0.0
        best_qz = 0.0
        best_t = 0.0

        for j in range(e):
            qx, qy, qz, t, d = _project_point_to_segment_scalar(
                px,
                py,
                pz,
                segment_starts[j, 0],
                segment_starts[j, 1],
                segment_starts[j, 2],
                segment_ends[j, 0],
                segment_ends[j, 1],
                segment_ends[j, 2],
            )
            if d < best_d:
                best_d = d
                best_e = j
                best_qx = qx
                best_qy = qy
                best_qz = qz
                best_t = t

        nearest_edge[i] = best_e
        nearest_point[i, 0] = best_qx
        nearest_point[i, 1] = best_qy
        nearest_point[i, 2] = best_qz
        distance[i] = best_d
        t_out[i] = best_t

    return nearest_edge, nearest_point, distance, t_out


@njit(**_JIT)
def project_points_to_candidate_segments(
    points,
    segment_starts,
    segment_ends,
    candidate_edges,
    n_candidates,
):
    """Exact projection among per-point candidate edge indices."""
    n = points.shape[0]
    nearest_edge = empty(n, dtype=int64)
    nearest_point = empty((n, 3), dtype=float64)
    distance = empty(n, dtype=float64)
    t_out = empty(n, dtype=float64)

    for i in range(n):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        best_d = inf
        best_e = 0
        best_qx = 0.0
        best_qy = 0.0
        best_qz = 0.0
        best_t = 0.0

        for kk in range(n_candidates[i]):
            j = candidate_edges[i, kk]
            if j < 0:
                continue
            qx, qy, qz, t, d = _project_point_to_segment_scalar(
                px,
                py,
                pz,
                segment_starts[j, 0],
                segment_starts[j, 1],
                segment_starts[j, 2],
                segment_ends[j, 0],
                segment_ends[j, 1],
                segment_ends[j, 2],
            )
            if d < best_d:
                best_d = d
                best_e = j
                best_qx = qx
                best_qy = qy
                best_qz = qz
                best_t = t

        nearest_edge[i] = best_e
        nearest_point[i, 0] = best_qx
        nearest_point[i, 1] = best_qy
        nearest_point[i, 2] = best_qz
        distance[i] = best_d
        t_out[i] = best_t

    return nearest_edge, nearest_point, distance, t_out


def _project_midpoint_kdtree(pts, starts, ends, k_candidates: int):
    """Candidate search via edge midpoints, then exact segment projection.

    Not guaranteed exhaustive unless ``k_candidates >= E``. Prefer
    ``method=\"bruteforce\"`` when a guaranteed mapping is required.
    """
    from scipy.spatial import cKDTree

    n = pts.shape[0]
    e = starts.shape[0]
    mid = 0.5 * (starts + ends)
    tree = cKDTree(mid)
    k = int(min(max(int(k_candidates), 1), e))
    _, idx = tree.query(pts, k=k)
    if k == 1:
        idx = idx.reshape(n, 1)
    cand = asarray(idx, dtype=int64)
    n_cand = empty(n, dtype=int64)
    n_cand[:] = k
    return project_points_to_candidate_segments(pts, starts, ends, cand, n_cand)


def project_points_to_segments(
    points,
    segment_starts,
    segment_ends,
    *,
    method: str = "bruteforce",
    k_candidates: int = 64,
):
    """Project points onto the nearest line segments.

    Parameters
    ----------
    points : array-like, shape (N, 3)
    segment_starts, segment_ends : array-like, shape (E, 3)
    method : {\"bruteforce\", \"midpoint_kdtree\", \"auto\"}
        ``bruteforce`` is the reference (exact, O(N·E)).
        ``midpoint_kdtree`` searches ``k_candidates`` nearest edge midpoints
        then projects exactly onto those candidates (fast, not guaranteed).
        ``auto`` uses bruteforce when ``N·E`` is modest, else midpoint KD-tree.
    k_candidates : int
        Neighbours retained by the KD-tree candidate stage.

    Returns
    -------
    nearest_edge : (N,) int64
    nearest_point : (N, 3) float64
    distance : (N,) float64
    t : (N,) float64
        Edge fraction in ``[0, 1]`` (0 = start / source, 1 = end / target).
    """
    pts = asarray(points, dtype=float64)
    starts = asarray(segment_starts, dtype=float64)
    ends = asarray(segment_ends, dtype=float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 3:
        raise ValueError("segment_starts/ends must both have shape (E, 3)")
    if starts.shape[0] == 0:
        raise ValueError("Cannot project onto an empty segment set")

    n = pts.shape[0]
    e = starts.shape[0]
    if n == 0:
        return (
            empty(0, dtype=int64),
            empty((0, 3), dtype=float64),
            empty(0, dtype=float64),
            empty(0, dtype=float64),
        )

    if method not in ("bruteforce", "midpoint_kdtree", "auto"):
        raise ValueError(f"Unknown method {method!r}")

    if method == "bruteforce" or (method == "auto" and n * e <= 5_000_000):
        return project_points_to_segments_bruteforce(pts, starts, ends)

    return _project_midpoint_kdtree(pts, starts, ends, k_candidates)
