"""Neuron Mesh class.

This module provides the Tree_mesh class for representing neuron morphologies
as surface meshes.
"""

from vedo import Mesh

from ..core import _Mesh


class Tree_mesh(_Mesh):
    """Neuron mesh class.

    Parameters
    ----------
    ID : int | str
        Unique identifier for the neuron.
    metadata : dict
        Metadata dictionary containing information about the neuron.
    mesh : vedo.Mesh
        The vedo mesh object representing the neuron surface.
    """

    __slots__ = ()

    def __init__(self, ID: int | str, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID=ID, metadata=metadata, mesh=mesh)
