"""Neuron Mesh class """
from vedo import Mesh

from ..core import _Mesh

class Tree_mesh(_Mesh):
    """ Neuron mesh class """
    def __init__(self, ID: int, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID = ID, metadata = metadata, mesh = mesh)
