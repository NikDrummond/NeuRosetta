""""""

from ..core import _Forest

class Neuropils(_Forest):
    """API collection of neuropil meshes """
    def __init__(self, meshes):
        super().__init__(trees = meshes)
