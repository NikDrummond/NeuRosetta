from numpy.linalg import norm

from ..core import _Tree
from ..graphs.properties import _bind_edge_property

# distance between edges

def edge_length(tree: _Tree, bind: bool = True):
    """_summary_

    Parameters
    ----------
    tree : _Tree
        _description_
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    _type_
        _description_
    """
    starts, stops = tree.get_edge_coordinates()
    lengths = norm(starts - stops, axis = 1)
    if bind:
        _bind_edge_property(tree.graph, 'path_length', 'double', lengths)
    else:
        return lengths
