"""NumPy helper utilities for distance computations."""

from numpy.linalg import norm
from numpy import ndarray


def pairwise_distance(a: ndarray, b: ndarray) -> ndarray:
    """Compute pairwise Euclidean distance between two arrays of points.

    Parameters
    ----------
    a : ndarray
        First array of points with shape (N, D) where N is the number of points
        and D is the dimensionality.
    b : ndarray
        Second array of points with shape (N, D). Must have the same shape as `a`.

    Returns
    -------
    ndarray
        Array of Euclidean distances with shape (N,), where each element is
        the distance between corresponding rows of `a` and `b`.

    Examples
    --------
    >>> a = np.array([[0, 0], [1, 1]])
    >>> b = np.array([[1, 0], [2, 2]])
    >>> pairwise_distance(a, b)
    array([1., 1.414...])
    """
    return norm(a - b, axis=1)