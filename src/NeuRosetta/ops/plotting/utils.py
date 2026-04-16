from numpy.random import choice
from numpy import ndarray
from vedo import Lines, Point

from ...core import _Tree


def _vd_tree_lines(tree: _Tree, kwargs: dict) -> Lines:
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

    r_coords = tree.get_node_coordinates(subset=tree.root_index())
    pnt = Point(r_coords, **kwargs)
    return pnt

def _random_c() -> ndarray:
    return list(choice(range(256), size = 3))

def _build_3d(
    tree: _Tree,
    cache: bool = True,
    line_kwargs: dict | None= None,
    root_kwargs: dict | None = None,
    random_c: bool = False,
) -> dict | None:

    if hasattr(tree, "_plot_dict"):
        return

    if random_c:
        c = _random_c()
    else:
        c = 'k'
    
    if line_kwargs is None:
        line_kwargs = {"c": c, "lw": 1, "alpha": 1.0}
    if root_kwargs is None:
        root_kwargs = {"r": 12, "c": c, "alpha": 1.0}
    
    plot_dict = {
        "lns": _vd_tree_lines(tree, line_kwargs),
        "root": _vd_tree_root(tree, root_kwargs),
    }

    if cache:
        tree._plot_dict = plot_dict
    else:
        return plot_dict

