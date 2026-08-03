"""Internal helpers for global forest operations."""
from numpy import cumsum, split, ndarray

from ...core import _Forest

def _split_inds(forest:_Forest) -> ndarray:
    """ Given a forest, return splitting indices to split an array into individual trees"""
    counts = forest.count_nodes(parallel = True, show_progress = False)
    return cumsum(counts)[:-1]

def _split_array_vertex(arr: ndarray, inds: ndarray) -> ndarray:
    """Given an array equal to the number of nodes in all trees in a forest, split it into an array for each tree"""
    return split(arr, inds)