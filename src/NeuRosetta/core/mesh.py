""" base mesh class """
from vedo import Mesh

from .stone import _Stone

class _Mesh(_Stone):
    """Core mesh class"""
    def __init__(self, ID:int | str, metadata: dict, mesh: Mesh) -> None:
        super().__init__(ID, metadata)
        self.mesh = mesh

