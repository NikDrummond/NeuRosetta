"""pojections - TODO move some basic operations from algebra to here"""

from numba import njit
from numpy import empty

from .algebra import project_scalar


_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")

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


# ------------------------------------------------------------------
# Reflection about axis
# ------------------------------------------------------------------

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