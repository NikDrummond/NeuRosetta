""" Functions for global coordinate transformation of forests"""

from numpy import array, ndarray

from ...core import _Forest
from ...utils.graph_utils.gt_properties import _set_coords_prop
from .utils import _split_array_vertex, _split_inds

def _set_global_coords(forest: _Forest, x:ndarray,y:ndarray,z:ndarray):
    s_index = _split_inds(forest)

    x = _split_array_vertex(x, s_index)
    y = _split_array_vertex(y, s_index)
    z = _split_array_vertex(z, s_index)

    for i in range(len(forest)):
        c = array([x[i],y[i],z[i]])
        _set_coords_prop(forest[i].graph, c)