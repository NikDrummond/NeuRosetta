""" Forest container for multiple neuron meshes (API)"""

from ..core import _Forest

class Forest_mesh(_Forest):
    """API collection of neuron meshes """
    def __init__(self, meshes):
        super().__init__(trees = meshes)
