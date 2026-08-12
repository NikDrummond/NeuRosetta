"""pojections - TODO move some basic operations from algebra to here"""

from numba import njit
from numpy import empty, float64

from .algebra import project_scalar, normalize_scalar
from ._validation import _check_scalar, _check_vector_broadcast, _broadcast_vectors


_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")

# scalar

@njit(**_JIT)
def reflect_plane_scalar(
    x, y, z,
    nx, ny, nz,
):

    px, py, pz = project_scalar(
        x, y, z,
        nx, ny, nz,
    )

    return (
        x - 2.0*px,
        y - 2.0*py,
        z - 2.0*pz,
    )

@njit(**_JIT)
def mirror_axis_scalar(
    x, y, z,
    ax, ay, az,
):

    px, py, pz = project_scalar(
        x, y, z,
        ax, ay, az,
    )

    return (
        2.0*px - x,
        2.0*py - y,
        2.0*pz - z,
    )

@njit(**_JIT)
def bisector_scalar(
    x1,y1,z1,
    x2,y2,z2,
):
    """
    Return the unit vector bisecting two vectors.

    The input vectors do not need to be normalized.
    """
    x1,y1,z1 = normalize_scalar(x1,y1,z1)
    x2,y2,z2 = normalize_scalar(x2,y2,z2)

    bx = x1 + x2
    by = y1 + y2
    bz = z1 + z2

    return normalize_scalar(bx,by,bz)


# array

@njit(**_JIT)
def reflect_plane_xyz(
    x, y, z,
    nx, ny, nz,
):

    n = x.size

    xo = empty(n)
    yo = empty(n)
    zo = empty(n)

    for i in range(n):

        xo[i], yo[i], zo[i] = reflect_plane_scalar(
            x[i], y[i], z[i],
            nx[i], ny[i], nz[i],
        )

    return xo, yo, zo

@njit(**_JIT)
def mirror_axis_xyz(
    x, y, z,
    ax, ay, az,
):

    n = x.size

    xo = empty(n)
    yo = empty(n)
    zo = empty(n)

    for i in range(n):

        xo[i], yo[i], zo[i] = mirror_axis_scalar(
            x[i], y[i], z[i],
            ax[i], ay[i], az[i],
        )

    return xo, yo, zo

@njit(**_JIT)
def bisector_xyz(
    x1, y1, z1,
    x2, y2, z2,
):
    """
    Return unit bisectors for arrays of vectors.
    """

    n = x1.size

    bx = empty(n, dtype=float64)
    by = empty(n, dtype=float64)
    bz = empty(n, dtype=float64)

    for i in range(n):

        bx[i], by[i], bz[i] = bisector_scalar(
            x1[i], y1[i], z1[i],
            x2[i], y2[i], z2[i],
        )

    return bx, by, bz

# python wrappers

def bisector(x1, y1, z1, x2, y2, z2):
    """
    Unit vector bisecting two 3D vectors.

    Each input vector is normalized before computing the bisector. The result
    is the normalized sum of the two unit vectors.

    Parameters
    ----------
    x1, y1, z1 : float or ndarray
        Components of the first vector.
    x2, y2, z2 : float or ndarray
        Components of the second vector. A scalar vector may be broadcast
        against an array vector.

    Returns
    -------
    bx, by, bz : float or ndarray
        Components of the unit bisector vector.
    """
    if _check_vector_broadcast(x1, y1, z1, x2, y2, z2):
        x1, y1, z1, x2, y2, z2 = _broadcast_vectors(x1, y1, z1, x2, y2, z2)

    if _check_scalar(x1, y1, z1):
        return bisector_scalar(x1, y1, z1, x2, y2, z2)
    return bisector_xyz(x1, y1, z1, x2, y2, z2)


def reflect_plane(x, y, z, nx, ny, nz):
    """
    Reflect a 3D vector across a plane.

    The plane is defined by its normal ``(nx, ny, nz)``. The reflection is
    ``v - 2 * project(v, n)``.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector to reflect.
    nx, ny, nz : float or ndarray
        Components of the plane normal. A scalar normal may be broadcast
        against an array vector.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components of the reflected vector.
    """
    if _check_vector_broadcast(x, y, z, nx, ny, nz):
        x, y, z, nx, ny, nz = _broadcast_vectors(x, y, z, nx, ny, nz)

    if _check_scalar(x, y, z):
        return reflect_plane_scalar(x, y, z, nx, ny, nz)
    return reflect_plane_xyz(x, y, z, nx, ny, nz)


def mirror_axis(x, y, z, ax, ay, az):
    """
    Mirror a 3D vector across an axis.

    The mirror transformation is ``2 * project(v, axis) - v``, which reflects
    the component orthogonal to the axis while preserving the component along
    the axis.

    Parameters
    ----------
    x, y, z : float or ndarray
        Components of the vector to mirror.
    ax, ay, az : float or ndarray
        Components of the mirror axis. A scalar axis may be broadcast against
        an array vector.

    Returns
    -------
    xo, yo, zo : float or ndarray
        Components of the mirrored vector.
    """
    if _check_vector_broadcast(x, y, z, ax, ay, az):
        x, y, z, ax, ay, az = _broadcast_vectors(x, y, z, ax, ay, az)

    if _check_scalar(x, y, z):
        return mirror_axis_scalar(x, y, z, ax, ay, az)
    return mirror_axis_xyz(x, y, z, ax, ay, az)
