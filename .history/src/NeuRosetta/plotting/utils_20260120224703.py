from vedo import Lines, Point

from ..core import _Tree

def _vd_tree_lines(tree:_Tree, kwargs: dict) -> Lines:
    """generate a vedo.Lines object for a neuron

    Parameters
    ----------
    tree : nr.Tree_graph
        Neuron tree
    kwargs : dict, optional
        kwargs passed to vedo.Lines. 
        If nothing is passed, uses {"c":"k4", "lw":1,"alpha":1.0} by default.

    Returns
    -------
    vd.Lines
        Line representation of neuron for plotting.
    """
    starts, stops = tree.get_edge_coordinates()
    lns = Lines(starts, stops, **kwargs)

    return lns

def _vd_tree_root(tree: _Tree, kwargs: dict) -> Point:

    r_coords = tree.get_node_coordinates(subset = tree.root_index())
    pnt = Point(r_coords, **kwargs)
    return pnt

def _build_3d(tree: _Tree, 
            cache: bool = True,
            line_kwargs: dict = {"c":"k4", "lw":1, "alpha":1.0}, 
            root_kwargs: dict = {"r":12, "c": 'k4', "alpha":1.0}) -> dict | None:
    
    plot_dict = {'lns'_vd_tree_lines(tree, line_kwargs)
    root = _vd_tree_root(tree, root_kwargs)

    if cache:
        tree._plot_dict = {'lns':lns,'root':root}
    else:
        return 

