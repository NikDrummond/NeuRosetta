from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property

def vertex_coordinates(tree:_Tree, subset : int | List | bool = None):
    """
    return spatial coordinates of vertices in an np.array
    """

    # return an np.array of coordinates from a graph

    _check_internal_property(tree.graph,'coordinates')

    coords = tree.graph.vp['coordinates'].get_2d_array().T

    if subset is not None:
        coords = coords[subset]

    return coords