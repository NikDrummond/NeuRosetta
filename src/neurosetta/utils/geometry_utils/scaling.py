from numba import njit
from numpy import empty, float64, ndarray

from .algebra import project_scalar

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def scale_along_basis_scalar(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    bx1,
    by1,
    bz1,
    bx2,
    by2,
    bz2,
    bx3,
    by3,
    bz3,
):
    """
    Scale a vector along an orthonormal basis.

    Assumes the basis vectors are normalized.
    """

    px1, py1, pz1 = project_scalar(
        x,
        y,
        z,
        bx1,
        by1,
        bz1,
    )

    px2, py2, pz2 = project_scalar(
        x,
        y,
        z,
        bx2,
        by2,
        bz2,
    )

    px3, py3, pz3 = project_scalar(
        x,
        y,
        z,
        bx3,
        by3,
        bz3,
    )

    return (
        sx * px1 + sy * px2 + sz * px3,
        sx * py1 + sy * py2 + sz * py3,
        sx * pz1 + sy * pz2 + sz * pz3,
    )


@njit(**_JIT)
def scale_along_basis_xyz(
    x,
    y,
    z,
    sx,
    sy,
    sz,
    bx1,
    by1,
    bz1,
    bx2,
    by2,
    bz2,
    bx3,
    by3,
    bz3,
):
    """
    Scale an array of vectors along an orthonormal basis.

    Assumes the basis vectors are normalized.
    """

    n = x.size

    xo = empty(n, dtype=float64)
    yo = empty(n, dtype=float64)
    zo = empty(n, dtype=float64)

    for i in range(n):
        xo[i], yo[i], zo[i] = scale_along_basis_scalar(
            x[i],
            y[i],
            z[i],
            sx,
            sy,
            sz,
            bx1,
            by1,
            bz1,
            bx2,
            by2,
            bz2,
            bx3,
            by3,
            bz3,
        )

    return xo, yo, zo


def scale_along_basis(
    x: ndarray,
    y: ndarray,
    z: ndarray,
    s: float | int | ndarray,
    b1: ndarray,
    b2: ndarray,
    b3: ndarray,
):
    """
    Scale an array of 3D vectors along an orthonormal basis.

    Each vector is decomposed into its projections onto ``b1``, ``b2``, and
    ``b3``, then reassembled with independent scale factors applied along each
    basis direction.

    Parameters
    ----------
    x, y, z : ndarray
        1D coordinate arrays of equal length.
    s : float, int, or ndarray
        Scale factor(s) along ``b1``, ``b2``, and ``b3``. A scalar is applied
        uniformly to all three directions.
    b1, b2, b3 : ndarray
        Three orthonormal basis vectors, each with shape ``(3,)``.

    Returns
    -------
    xo, yo, zo : ndarray
        Components of the scaled vectors.

    Notes
    -----
    The basis vectors are assumed to be normalized.
    """
    if isinstance(s, (float, int)):
        s = [s, s, s]
    return scale_along_basis_xyz(x, y, z, *s, *b1, *b2, *b3)
