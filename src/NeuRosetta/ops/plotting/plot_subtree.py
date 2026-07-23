"""Get lines to plot subtree."""

from typing import Tuple
from numpy import array
from vedo import Lines

from ...core import _Tree
from ..tree_graphs import tree_has_property


def build_3d_subtree(
    tree: _Tree,
    c1: str = "g",
    c2: str = "r",
    lw1: float = 2,
    lw2: float = 2,
    **kwargs,
) -> Tuple[Lines, Lines]:
    """Build vedo Lines objects for a subtree and its complement.

    Parameters
    ----------
    tree : _Tree
        Neuron tree with subtree masks.
    c1 : str, optional
        Color for lines inside the subtree. By default "g" (green).
    c2 : str, optional
        Color for lines outside the subtree. By default "r" (red).
    lw1 : float, optional
        Line width for lines inside the subtree. By default 2.
    lw2 : float, optional
        Line width for lines outside the subtree. By default 2.
    **kwargs : dict
        Additional keyword arguments passed to vedo.Lines.

    Returns
    -------
    Tuple[Lines, Lines]
        Tuple of (lines_in_subtree, lines_outside_subtree) as vedo.Lines objects.

    Raises
    ------
    AttributeError
        If the tree does not have bound subtree masks for vertices and edges.
    """
    if tree_has_property(tree, "v_subtree_mask", "v") & tree_has_property(
        tree, "e_subtree_mask", "e"
    ):

        # get coordinates, edges, and masks
        coords = tree.get_node_coordinates()
        edges = tree.edge_indices()
        mask = array(tree.graph.ep["e_subtree_mask"].a, dtype=bool)

        # get lines in subtree
        in_edges = edges[mask]
        starts = coords[in_edges[:, 0]]
        stops = coords[in_edges[:, 1]]
        lns_in = Lines(starts, stops, c=c1, lw=lw1, **kwargs)

        # lines not in the subtree
        out_edges = edges[~mask]
        starts = coords[out_edges[:, 0]]
        stops = coords[out_edges[:, 1]]
        lns_out = Lines(starts, stops, c=c2, lw=lw2, **kwargs)

        # return
        return lns_in, lns_out

    # if we don't have the needed properties raise
    raise AttributeError(
        "Neuron must have bound subtree masks for vertices and edges"
    )