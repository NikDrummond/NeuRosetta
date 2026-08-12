"""Internal helpers for global forest operations."""
from numpy import cumsum, split, ndarray

from ...core import _Forest

def _split_inds(forest:_Forest) -> ndarray:
    """Return split indices for partitioning a pooled vertex array by tree.

    Parameters
    ----------
    forest : _Forest
        Forest whose node counts define the split boundaries.

    Returns
    -------
    ndarray
        Cumulative split indices excluding the final boundary.
    """
    counts = forest.count_nodes(parallel = True, show_progress = False)
    return cumsum(counts)[:-1]

def _split_array_vertex(arr: ndarray, inds: ndarray) -> ndarray:
    """Split a pooled vertex array into per-tree segments.

    Parameters
    ----------
    arr : ndarray
        Array with length equal to the total number of nodes in the forest.
    inds : ndarray
        Split indices from :func:`_split_inds`.

    Returns
    -------
    ndarray
        Array of per-tree segments.
    """
    return split(arr, inds)