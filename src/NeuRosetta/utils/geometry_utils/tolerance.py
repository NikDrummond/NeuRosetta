# tolerance.py

import numpy as np

from .algebra import magnitude


def almost_zero(v, atol=1e-08):
    """
    Test whether a 3D vector is close to the origin.

    Parameters
    ----------
    v : array-like
        3D vector.
    atol : float, default 1e-8
        Absolute tolerance for each component.

    Returns
    -------
    bool
        ``True`` if all components are within ``atol`` of zero.
    """
    return np.allclose(v, np.array([0.0, 0.0, 0.0]), rtol=0, atol=atol)


def almost_unit_length(vector, atol=1e-08):
    """
    Test whether a 3D vector has unit length.

    Parameters
    ----------
    vector : array-like
        3D vector.
    atol : float, default 1e-8
        Absolute tolerance on ``|vector| - 1``.

    Returns
    -------
    bool
        ``True`` if the magnitude is within ``atol`` of 1.
    """
    vector = np.asarray(vector, dtype=float)
    return np.isclose(magnitude(vector[0], vector[1], vector[2]), 1.0, rtol=0, atol=atol)


def almost_collinear(v1, v2, atol=1e-08):
    """
    Test whether two 3D vectors are nearly collinear.

    Parameters
    ----------
    v1, v2 : array-like
        3D vectors.
    atol : float, default 1e-8
        Absolute tolerance on ``|v1 x v2|``.

    Returns
    -------
    bool
        ``True`` if the cross-product norm is within ``atol`` of zero.
    """
    cross_product = np.cross(v1, v2)
    norm = np.linalg.norm(cross_product)
    return np.isclose(norm, 0.0, rtol=0, atol=atol)


def almost_equal(v1, v2, atol=1e-08):
    """
    Test whether two 3D vectors are nearly equal.

    Parameters
    ----------
    v1, v2 : array-like
        3D vectors to compare.
    atol : float, default 1e-8
        Absolute tolerance for each component.

    Returns
    -------
    bool
        ``True`` if all corresponding components differ by at most ``atol``.
    """
    return np.allclose(v1, v2, rtol=0, atol=atol)