"""Neuropil mesh class.

This module provides the Neuropil class for representing brain region
boundaries as surface meshes.
"""

from vedo import Mesh

from ..core import _Mesh


class Neuropil(_Mesh):
    """Neuropil mesh class.

    Parameters
    ----------
    ID : str
        Unique identifier for the neuropil region.
    metadata : dict
        Metadata dictionary containing information about the neuropil.
    mesh : vedo.Mesh
        The vedo mesh object representing the neuropil surface.
    """

    __slots__ = ()

    def __init__(self, ID: str, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID=ID, metadata=metadata, mesh=mesh)
