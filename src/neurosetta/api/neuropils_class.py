"""Neuropil container class.

This module provides the Neuropils class for managing collections of Neuropil
objects.
"""

from ..core import _Forest


class Neuropils(_Forest):
    """API collection of neuropil meshes.

    Parameters
    ----------
    meshes : list[Neuropil]
        List of Neuropil objects to include in the collection.
    """

    __slots__ = ()

    def __init__(self, meshes):
        super().__init__(trees=meshes)