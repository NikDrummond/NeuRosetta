import vedo as vd

vd.settings.default_backend = "vtk"

def _vd_tree_lines(tree:nr.Tree_graph, kwargs: dict) -> vd.Lines:
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
    lns = vd.Lines(starts, stops, **kwargs)

    return lns

def _vd_tree_root(tree, kwargs: dict) -> vd.Point:

    r_coords = tree.get_node_coordinates(subset = tree.root_index())
    pnt = vd.Point(r_coords, **kwargs)
    return pnt

def plot_3d(tree: nr.Tree_graph, 
            line_kwargs:dict = {"c":"k4", "lw":1,"alpha":1.0}, 
            root_kwargs: dict = {"r":12, "c": 'k4', "alpha":1.0},
            plot_kwargs: dict = {"axes": None}) -> vd.Plotter:
    """On the fly 3D neuron plotting. Opens an interactive vedo.Plotter instance with the neuron.

    Parameters
    ----------
    tree : nr.Tree_graph
        Neuron tree
    line_kwargs : dict, optional
        keyword arguments passed to vedo.Lines to customise the neuron.
        By default {"c":"k4", "lw":1,"alpha":1.0}
    root_kwargs : dict, optional
        keyword arguments passed to vedo.Point to customise how the soma is plotted.
        By default {"r":12, "c": 'k4', "alpha":1.0}
    plot_kwargs : dict, optional
        Keyword arguments passed to vedo.show. 
        By default {"axes": None}

    Returns
    -------
    vd.Plotter
        _description_
    """
    lns = _vd_tree_lines(tree, line_kwargs)
    pnt = _vd_tree_root(tree, root_kwargs)

    vd.show([lns, pnt], **plot_kwargs).close()
