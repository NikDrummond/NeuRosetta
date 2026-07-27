"""NumPy helper utilities for distance computations."""

from numpy import ndarray
from numpy.linalg import norm


def euclidean_distance(a: ndarray, b: ndarray) -> ndarray:
    """Compute pairwise Euclidean distance between two arrays of points.

    Parameters
    ----------
    a : ndarray
        First array of points with shape ``(N, D)`` where ``N`` is the number
        of points and ``D`` is the dimensionality.
    b : ndarray
        Second array of points with shape ``(N, D)``. Must have the same shape
        as ``a``.

    Returns
    -------
    ndarray
        Array of Euclidean distances with shape ``(N,)``, where each element is
        the distance between corresponding rows of ``a`` and ``b``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[0, 0], [1, 1]])
    >>> b = np.array([[1, 0], [2, 2]])
    >>> euclidean_distance(a, b)
    array([1.        , 1.41421356])
    """
    return norm(a - b, axis=1)
