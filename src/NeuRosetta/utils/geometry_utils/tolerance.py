# tolerance.py

import numpy as np

from .algebra import magnitude


def almost_zero(v, atol=1e-08):
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_unit_length(vector, atol=1e-08):
    return np.isclose(magnitude(vector), 1.0, rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    cross_product = np.cross(v1, v2)
    norm = np.linalg.norm(cross_product)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def almost_equal(v1, v2, atol=1e-08):
    return np.allclose(v1, v2, rtol=0, atol=atol)