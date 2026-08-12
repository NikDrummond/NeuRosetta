# basis.py

import numpy as np
from numba import njit

from .algebra import dot_scalar 

# Precomputed once, at import time, instead of allocating a fresh array on
# every `basis.x` / `basis.y` / ... access. Marked read-only so accidental
# in-place mutation (e.g. `basis.x[0] = 5`) fails loudly instead of silently
# corrupting a value shared across every caller.
#
# If you need a *mutable* copy, call `.copy()` on the returned array.

_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
_Z = np.array([0.0, 0.0, 1.0])
_NEG_X = np.array([-1.0, 0.0, 0.0])
_NEG_Y = np.array([0.0, -1.0, 0.0])
_NEG_Z = np.array([0.0, 0.0, -1.0])

for _arr in (_X, _Y, _Z, _NEG_X, _NEG_Y, _NEG_Z):
    _arr.flags.writeable = False


class _BasisVectors(object):
    """
    The cartesian basis vectors. Returned arrays are shared, read-only views;
    call `.copy()` if you need a mutable array of your own.
    """

    @property
    def x(self):
        return _X

    @property
    def y(self):
        return _Y

    @property
    def z(self):
        return _Z

    @property
    def neg_x(self):
        return _NEG_X

    @property
    def neg_y(self):
        return _NEG_Y

    @property
    def neg_z(self):
        return _NEG_Z


basis = _BasisVectors()

# basis decomposition

_JIT = dict(nogil=True, fastmath=True, cache=True, inline="always")


@njit(**_JIT)
def decompose_basis_scalar(
    x, y, z,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return (
        dot_scalar(x, y, z, bx1, by1, bz1),
        dot_scalar(x, y, z, bx2, by2, bz2),
        dot_scalar(x, y, z, bx3, by3, bz3),
    )


@njit(**_JIT)
def decompose_basis_xyz(
    x, y, z,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    n = x.size

    c1 = np.empty(n)
    c2 = np.empty(n)
    c3 = np.empty(n)

    for i in range(n):

        c1[i], c2[i], c3[i] = decompose_basis_scalar(
            x[i], y[i], z[i],
            bx1, by1, bz1,
            bx2, by2, bz2,
            bx3, by3, bz3,
        )

    return c1, c2, c3

# basis reconstuction


@njit(**_JIT)
def reconstruct_basis_scalar(
    c1, c2, c3,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return (
        c1*bx1 + c2*bx2 + c3*bx3,
        c1*by1 + c2*by2 + c3*by3,
        c1*bz1 + c2*bz2 + c3*bz3,
    )


@njit(**_JIT)
def reconstruct_basis_xyz(
    c1, c2, c3,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    n = c1.size

    x = np.empty(n)
    y = np.empty(n)
    z = np.empty(n)

    for i in range(n):

        x[i], y[i], z[i] = reconstruct_basis_scalar(
            c1[i], c2[i], c3[i],
            bx1, by1, bz1,
            bx2, by2, bz2,
            bx3, by3, bz3,
        )

    return x, y, z


# ------------------------------------------------------------------
# Change of basis
# ------------------------------------------------------------------

@njit(**_JIT)
def change_basis_scalar(
    x, y, z,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return decompose_basis_scalar(
        x, y, z,
        bx1, by1, bz1,
        bx2, by2, bz2,
        bx3, by3, bz3,
    )


@njit(**_JIT)
def change_basis_xyz(
    x, y, z,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return decompose_basis_xyz(
        x, y, z,
        bx1, by1, bz1,
        bx2, by2, bz2,
        bx3, by3, bz3,
    )


# ------------------------------------------------------------------
# Inverse change of basis
# ------------------------------------------------------------------

@njit(**_JIT)
def inverse_change_basis_scalar(
    c1, c2, c3,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return reconstruct_basis_scalar(
        c1, c2, c3,
        bx1, by1, bz1,
        bx2, by2, bz2,
        bx3, by3, bz3,
    )


@njit(**_JIT)
def inverse_change_basis_xyz(
    c1, c2, c3,
    bx1, by1, bz1,
    bx2, by2, bz2,
    bx3, by3, bz3,
):

    return reconstruct_basis_xyz(
        c1, c2, c3,
        bx1, by1, bz1,
        bx2, by2, bz2,
        bx3, by3, bz3,
    )