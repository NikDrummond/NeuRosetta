# point.py

from numba import njit
from numpy import sqrt, empty, float64, bool_

from ._validation import _check_vector_broadcast, _check_scalar, _broadcast_vectors

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")

### Numbar

### Scalar kernels


@njit(**_JIT)
def euclidean_distance_scalar(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
):
    """
    Euclidean distance between two 3D points.
    """

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    return sqrt(dx * dx + dy * dy + dz * dz)


### Array Kernels
@njit(**_JIT)
def euclidean_distance_xyz(
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
):

    n = x1.size

    out = empty(n, dtype=float64)

    for i in range(n):

        out[i] = euclidean_distance_scalar(
            x1[i],
            y1[i],
            z1[i],
            x2[i],
            y2[i],
            z2[i],
        )

    return out


@njit(**_JIT)
def argapex_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
):
    """
    Index of point furthest along axis.
    """

    n = x.size

    if n == 0:
        return -1

    best = 0

    best_value = x[0] * ax + y[0] * ay + z[0] * az

    for i in range(1, n):

        value = x[i] * ax + y[i] * ay + z[i] * az

        if value > best_value:
            best_value = value
            best = i

    return best


@njit(**_JIT)
def apex_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
):

    i = argapex_xyz(x, y, z, ax, ay, az)

    return (
        x[i],
        y[i],
        z[i],
    )


@njit(**_JIT)
def apex_and_opposite_xyz(
    x,
    y,
    z,
    ax,
    ay,
    az,
):

    n = x.size

    if n == 0:
        return -1, -1

    imax = 0
    imin = 0

    vmax = x[0] * ax + y[0] * ay + z[0] * az
    vmin = vmax

    for i in range(1, n):

        v = x[i] * ax + y[i] * ay + z[i] * az

        if v > vmax:
            vmax = v
            imax = i

        if v < vmin:
            vmin = v
            imin = i

    return imax, imin


@njit(**_JIT)
def nearest_xyz(
    x,
    y,
    z,
    px,
    py,
    pz,
):

    n = x.size

    best = 0

    dx = x[0] - px
    dy = y[0] - py
    dz = z[0] - pz

    best_dist = dx * dx + dy * dy + dz * dz

    for i in range(1, n):

        dx = x[i] - px
        dy = y[i] - py
        dz = z[i] - pz

        d = dx * dx + dy * dy + dz * dz

        if d < best_dist:
            best_dist = d
            best = i

    return best


@njit(**_JIT)
def farthest_xyz(
    x,
    y,
    z,
    px,
    py,
    pz,
):

    n = x.size

    best = 0

    dx = x[0] - px
    dy = y[0] - py
    dz = z[0] - pz

    best_dist = dx * dx + dy * dy + dz * dz

    for i in range(1, n):

        dx = x[i] - px
        dy = y[i] - py
        dz = z[i] - pz

        d = dx * dx + dy * dy + dz * dz

        if d > best_dist:
            best_dist = d
            best = i

    return best


@njit(**_JIT)
def within_radius_xyz(
    x,
    y,
    z,
    px,
    py,
    pz,
    radius,
    atol=1e-8,
):

    n = x.size

    mask = empty(n, dtype=bool_)

    r2 = (radius + atol) ** 2

    for i in range(n):

        dx = x[i] - px
        dy = y[i] - py
        dz = z[i] - pz

        mask[i] = (dx * dx + dy * dy + dz * dz) < r2

    return mask


@njit(**_JIT)
def average_xyz(
    x,
    y,
    z,
):

    n = x.size

    sx = 0.0
    sy = 0.0
    sz = 0.0

    for i in range(n):

        sx += x[i]
        sy += y[i]
        sz += z[i]

    inv = 1.0 / n

    return (
        sx * inv,
        sy * inv,
        sz * inv,
    )


@njit(**_JIT)
def weighted_average_xyz(
    x,
    y,
    z,
    weights,
):

    n = x.size

    sx = 0.0
    sy = 0.0
    sz = 0.0
    sw = 0.0

    for i in range(n):

        w = weights[i]

        sx += w * x[i]
        sy += w * y[i]
        sz += w * z[i]

        sw += w

    inv = 1.0 / sw

    return (
        sx * inv,
        sy * inv,
        sz * inv,
        sw,
    )


### Numpy callers
def euclidean_distance(x1, y1, z1, x2, y2, z2):
    """
    Euclidean distance between two 3D points.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Coordinates of the first point.
    x2, y2, z2 : float or ndarray
        Coordinates of the second point. A scalar point may be broadcast
        against an array of points.

    Returns
    -------
    float or ndarray
        ``sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)`` for each pair
        of points.
    """

    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return euclidean_distance_scalar(x1, y1, z1, x2, y2, z2)

    return euclidean_distance_xyz(x1, y1, z1, x2, y2, z2)


def argapex(x, y, z, ax, ay, az):
    """
    Index of the point furthest along an axis.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    ax, ay, az : float or ndarray
        Components of the axis direction. A scalar axis is broadcast against
        the point arrays.

    Returns
    -------
    int
        Index of the point with the largest projection onto the axis.
        Returns ``-1`` if there are no points.
    """
    return argapex_xyz(x, y, z, ax, ay, az)


def apex(x, y, z, ax, ay, az):
    """
    Coordinates of the point furthest along an axis.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    ax, ay, az : float or ndarray
        Components of the axis direction. A scalar axis is broadcast against
        the point arrays.

    Returns
    -------
    ax_coord, ay_coord, az_coord : float
        Coordinates of the apex point.
    """
    return apex_xyz(x, y, z, ax, ay, az)


def apex_and_opposite(x, y, z, ax, ay, az):
    """
    Indices of the points furthest and least far along an axis.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    ax, ay, az : float or ndarray
        Components of the axis direction. A scalar axis is broadcast against
        the point arrays.

    Returns
    -------
    imax, imin : int
        Indices of the points with the largest and smallest projection onto
        the axis. Both are ``-1`` if there are no points.
    """
    return apex_and_opposite_xyz(x, y, z, ax, ay, az)


def nearest(x, y, z, px, py, pz):
    """
    Index of the point nearest to a reference point.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    px, py, pz : float
        Coordinates of the reference point.

    Returns
    -------
    int
        Index of the closest point in ``(x, y, z)``.
    """
    return nearest_xyz(x, y, z, px, py, pz)


def farthest(x, y, z, px, py, pz):
    """
    Index of the point farthest from a reference point.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    px, py, pz : float
        Coordinates of the reference point.

    Returns
    -------
    int
        Index of the farthest point in ``(x, y, z)``.
    """
    return farthest_xyz(x, y, z, px, py, pz)


def within_radius(x, y, z, px, py, pz, radius, atol=1e-8):
    """
    Test which points lie within a sphere of given radius.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    px, py, pz : float
        Coordinates of the sphere center.
    radius : float
        Sphere radius.
    atol : float, default 1e-8
        Absolute tolerance added to ``radius`` before squaring.

    Returns
    -------
    mask : ndarray of bool
        ``True`` for points whose squared distance to ``(px, py, pz)`` is
        less than ``(radius + atol)**2``.
    """
    return within_radius_xyz(x, y, z, px, py, pz, radius, atol)


def average(x, y, z, weights=None):
    """
    Average of 3D points, optionally weighted.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    weights : ndarray, optional
        1D array of weights with the same length as ``x``. If omitted, an
        unweighted mean is computed.

    Returns
    -------
    mx, my, mz : float
        Average coordinates.
    sum_weights : float, optional
        Present only when ``weights`` is supplied; total weight used in the
        average.
    """
    if weights is None:
        return average_xyz(x, y, z)
    return weighted_average_xyz(x, y, z, weights)
