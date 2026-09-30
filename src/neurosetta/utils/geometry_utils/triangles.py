"""Point-to-triangle projection primitives (Numba).

These kernels operate on plain NumPy arrays and must not import Mesh / Tree.
"""

from __future__ import annotations

from numba import njit
from numpy import asarray, empty, float64, inf, int64

_JIT = dict(nogil=True, fastmath=True, cache=True)


@njit(**_JIT)
def _project_point_to_triangle_scalar(
    px,
    py,
    pz,
    ax,
    ay,
    az,
    bx,
    by,
    bz,
    cx,
    cy,
    cz,
):
    """Closest point on triangle ABC to point P (Ericson RTCD §5.1.5).

    Returns ``qx, qy, qz, dist``.
    """
    abx = bx - ax
    aby = by - ay
    abz = bz - az
    acx = cx - ax
    acy = cy - ay
    acz = cz - az
    apx = px - ax
    apy = py - ay
    apz = pz - az

    d1 = abx * apx + aby * apy + abz * apz
    d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        dx = px - ax
        dy = py - ay
        dz = pz - az
        return ax, ay, az, (dx * dx + dy * dy + dz * dz) ** 0.5

    bpx = px - bx
    bpy = py - by
    bpz = pz - bz
    d3 = abx * bpx + aby * bpy + abz * bpz
    d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        dx = px - bx
        dy = py - by
        dz = pz - bz
        return bx, by, bz, (dx * dx + dy * dy + dz * dz) ** 0.5

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        qx = ax + v * abx
        qy = ay + v * aby
        qz = az + v * abz
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        return qx, qy, qz, (dx * dx + dy * dy + dz * dz) ** 0.5

    cpx = px - cx
    cpy = py - cy
    cpz = pz - cz
    d5 = abx * cpx + aby * cpy + abz * cpz
    d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        dx = px - cx
        dy = py - cy
        dz = pz - cz
        return cx, cy, cz, (dx * dx + dy * dy + dz * dz) ** 0.5

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        qx = ax + w * acx
        qy = ay + w * acy
        qz = az + w * acz
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        return qx, qy, qz, (dx * dx + dy * dy + dz * dz) ** 0.5

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        denom = (d4 - d3) + (d5 - d6)
        w = (d4 - d3) / denom
        qx = bx + w * (cx - bx)
        qy = by + w * (cy - by)
        qz = bz + w * (cz - bz)
        dx = px - qx
        dy = py - qy
        dz = pz - qz
        return qx, qy, qz, (dx * dx + dy * dy + dz * dz) ** 0.5

    denom = va + vb + vc
    v = vb / denom
    w = vc / denom
    qx = ax + abx * v + acx * w
    qy = ay + aby * v + acy * w
    qz = az + abz * v + acz * w
    dx = px - qx
    dy = py - qy
    dz = pz - qz
    return qx, qy, qz, (dx * dx + dy * dy + dz * dz) ** 0.5


@njit(**_JIT)
def project_points_to_triangles_bruteforce(points, tri_a, tri_b, tri_c):
    """Exact nearest-triangle projection for every point (O(N·F))."""
    n = points.shape[0]
    f = tri_a.shape[0]

    nearest_face = empty(n, dtype=int64)
    nearest_point = empty((n, 3), dtype=float64)
    distance = empty(n, dtype=float64)

    for i in range(n):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        best_d = inf
        best_f = 0
        best_qx = 0.0
        best_qy = 0.0
        best_qz = 0.0

        for j in range(f):
            qx, qy, qz, d = _project_point_to_triangle_scalar(
                px,
                py,
                pz,
                tri_a[j, 0],
                tri_a[j, 1],
                tri_a[j, 2],
                tri_b[j, 0],
                tri_b[j, 1],
                tri_b[j, 2],
                tri_c[j, 0],
                tri_c[j, 1],
                tri_c[j, 2],
            )
            if d < best_d:
                best_d = d
                best_f = j
                best_qx = qx
                best_qy = qy
                best_qz = qz

        nearest_face[i] = best_f
        nearest_point[i, 0] = best_qx
        nearest_point[i, 1] = best_qy
        nearest_point[i, 2] = best_qz
        distance[i] = best_d

    return nearest_face, nearest_point, distance


@njit(**_JIT)
def project_points_to_candidate_triangles(
    points,
    tri_a,
    tri_b,
    tri_c,
    candidate_faces,
    n_candidates,
):
    """Exact projection among per-point candidate face indices."""
    n = points.shape[0]
    nearest_face = empty(n, dtype=int64)
    nearest_point = empty((n, 3), dtype=float64)
    distance = empty(n, dtype=float64)

    for i in range(n):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]
        best_d = inf
        best_f = 0
        best_qx = 0.0
        best_qy = 0.0
        best_qz = 0.0

        for kk in range(n_candidates[i]):
            j = candidate_faces[i, kk]
            if j < 0:
                continue
            qx, qy, qz, d = _project_point_to_triangle_scalar(
                px,
                py,
                pz,
                tri_a[j, 0],
                tri_a[j, 1],
                tri_a[j, 2],
                tri_b[j, 0],
                tri_b[j, 1],
                tri_b[j, 2],
                tri_c[j, 0],
                tri_c[j, 1],
                tri_c[j, 2],
            )
            if d < best_d:
                best_d = d
                best_f = j
                best_qx = qx
                best_qy = qy
                best_qz = qz

        nearest_face[i] = best_f
        nearest_point[i, 0] = best_qx
        nearest_point[i, 1] = best_qy
        nearest_point[i, 2] = best_qz
        distance[i] = best_d

    return nearest_face, nearest_point, distance


def _project_centroid_kdtree(pts, tri_a, tri_b, tri_c, k_candidates: int):
    """Candidate search via face centroids, then exact triangle projection."""
    from scipy.spatial import cKDTree

    n = pts.shape[0]
    f = tri_a.shape[0]
    centroids = (tri_a + tri_b + tri_c) / 3.0
    tree = cKDTree(centroids)
    k = int(min(max(int(k_candidates), 1), f))
    _, idx = tree.query(pts, k=k)
    if k == 1:
        idx = idx.reshape(n, 1)
    cand = asarray(idx, dtype=int64)
    n_cand = empty(n, dtype=int64)
    n_cand[:] = k
    return project_points_to_candidate_triangles(pts, tri_a, tri_b, tri_c, cand, n_cand)


def project_points_to_triangles(
    points,
    tri_a,
    tri_b,
    tri_c,
    *,
    method: str = "bruteforce",
    k_candidates: int = 64,
):
    """Project points onto the nearest triangles.

    Parameters
    ----------
    points : array-like, shape (N, 3)
    tri_a, tri_b, tri_c : array-like, shape (F, 3)
        Triangle vertex coordinates.
    method : {\"bruteforce\", \"centroid_kdtree\", \"auto\"}
        ``bruteforce`` is exact O(N·F). ``centroid_kdtree`` searches
        ``k_candidates`` nearest face centroids then projects exactly
        (fast, not guaranteed). ``auto`` picks based on ``N·F``.
    k_candidates : int
        Neighbours retained by the KD-tree candidate stage.

    Returns
    -------
    nearest_face : (N,) int64
    nearest_point : (N, 3) float64
    distance : (N,) float64
    """
    pts = asarray(points, dtype=float64)
    a = asarray(tri_a, dtype=float64)
    b = asarray(tri_b, dtype=float64)
    c = asarray(tri_c, dtype=float64)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {pts.shape}")
    if a.shape != b.shape or a.shape != c.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("tri_a/b/c must all have shape (F, 3)")
    if a.shape[0] == 0:
        raise ValueError("Cannot project onto an empty triangle set")

    n = pts.shape[0]
    f = a.shape[0]
    if n == 0:
        return (
            empty(0, dtype=int64),
            empty((0, 3), dtype=float64),
            empty(0, dtype=float64),
        )

    if method not in ("bruteforce", "centroid_kdtree", "auto"):
        raise ValueError(f"Unknown method {method!r}")

    if method == "bruteforce" or (method == "auto" and n * f <= 5_000_000):
        return project_points_to_triangles_bruteforce(pts, a, b, c)

    return _project_centroid_kdtree(pts, a, b, c, k_candidates)
