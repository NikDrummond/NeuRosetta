### functions for distance calculations
from numpy.linalg import norm
from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils.properties import _bind_edge_property
from .coordinates import edge_coordinates
from .tree_checks import is_Reduced
from ...utils.numpy_utils.distances import pairwise_distance


def euclidean_edge_length(
    tree: _Tree, subset: str | None = None, bind: bool = True
) -> ndarray:
    """Get length of edges. If a non-reduced neuron is provided, this is the "path length" of each edge in the graph.
    If a reduced neuron is provided, this is the euclidean distance between nodes.

    Parameters
    ----------
    tree : Tree
        _description_
    subset : str | None, optional
        _description_, by default None
    bind : bool, optional
        _description_, by default True

    Returns
    -------
    ndarray
        _description_
    """

    pairs = edge_coordinates(tree, subset=subset)
    lengths = pairwise_distance(pairs[0],pairs[1])

    # we can only bind if a subset is not given
    if subset is None:
        if bind:
            # base property name on if this is a reduced graph
            if is_Reduced(tree):
                p_name = "Euclidean_length"
            else:
                p_name = "Path_length"
            _bind_edge_property(
                tree,
                property_name=p_name,
                property_dtype="double",
                property_data=lengths,
            )
        else:
            return lengths
    else:
        return lengths
