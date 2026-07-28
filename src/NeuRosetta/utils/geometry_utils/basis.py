# basis.py

import numpy as np

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