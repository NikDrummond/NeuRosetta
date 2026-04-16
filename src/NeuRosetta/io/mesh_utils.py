"""Simple stand in for laoding meshes for GUI - to be expanded"""

from vedo import load as mesh_load
from vedo import Mesh

def load_mesh(fpath: str, colour: str = 'gray', alpha: float = 0.3) -> Mesh:
    """Basic mesh loading for GUI, to be built out later"""
    mesh = mesh_load(fpath)
    mesh.alpha(alpha)
    mesh.c(colour)
    return mesh
