"""Forest container for multiple neuron meshes (API).

This module provides the Forest_mesh class for managing collections of
Tree_mesh objects.
"""

from ..core import _Forest


class Forest_mesh(_Forest):
    """API collection of neuron meshes.

    Parameters
    ----------
    meshes : list[Tree_mesh]
        List of Tree_mesh objects to include in the collection.
    """

    def __init__(self, meshes):
        super().__init__(trees=meshes)