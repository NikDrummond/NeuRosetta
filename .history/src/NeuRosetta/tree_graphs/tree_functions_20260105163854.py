from ..core import _Tree
from typing import List
from ..errors.errors import _check_internal_property

def vertex_coordinates(tree:_Tree, subset : int | List | bool = None):
    """Returns an n by 3 np.array of node coordinates within the neuron 

    Parameters
    ----------
    tree : Tree_graph
        euron representations
    subset : int | List | bool, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """

    _check_internal_property(tree.graph,'coordinates')

    coords = tree.graph.vp['coordinates'].get_2d_array().T

    if subset is not None:
        coords = coords[subset]

    return coords