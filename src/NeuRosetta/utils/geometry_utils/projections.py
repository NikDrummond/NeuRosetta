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

def bisector(x1,y1,z1,x2,y2,z2):
    """"""
    if _check_vector_broadcast(x1,y1,z1,x2,y2,z2):
        x1,y1,z1,x2,y2,z2 = _broadcast_vectors(x1,y1,z1,x2,y2,z2)

    if _check_scalar(x1,y1,z1):
        return bisector_scalar(x1,y1,z1,x2,y2,z2)
    return bisector_xyz(x1,y1,z1,x2,y2,z2)
