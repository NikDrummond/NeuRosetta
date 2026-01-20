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

def _vd_tree_root(tree:_Tree, kwargs: dict) -> Point:

    r_coords = tree.get_node_coordinates(subset = tree.root_index())
    pnt = Point(r_coords, **kwargs)
    return pnt