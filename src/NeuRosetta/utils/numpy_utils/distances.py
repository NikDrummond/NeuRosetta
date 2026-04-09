from numpy.linalg import norm
from numpy import ndarray

def pairwise_distance(a: ndarray,b: ndarray) -> ndarray:
    """pairwise (row) euclidean distance between two arrays of points
    """
    return norm(a-b, axis = 1)
