""" Neuropil meshes API"""

from vedo import Mesh

from ..core import _Mesh

class Neuropil(_Mesh):
    """ Neuropil mesh class """
    def __init__(self, ID: str, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID = ID, metadata = metadata, mesh = mesh)
