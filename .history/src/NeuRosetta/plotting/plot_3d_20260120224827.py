import vedo as vd
from ..core import _Tree
from .utils import _vd_tree_lines, _vd_tree_root, _build_3d
from .viewer import Viewer




def plot_3d(tree: _Tree, 
            line_kwargs:dict = {"c":"k4", "lw":1,"alpha":1.0}, 
            root_kwargs: dict = {"r":12, "c": 'k4', "alpha":1.0},
            plot_kwargs: dict = {"axes": None},
            cache:bool = True) -> vd.Plotter:
    """On the fly 3D neuron plotting. Opens an interactive vedo.Plotter instance with the neuron.

    Parameters
    ----------
    tree : nr.Tree_graph
        Neuron tree
    line_kwargs : dict, optional
        keyword arguments passed to vedo.Lines to customise the neuron.
        By default {"c":"k4", "lw":1, "alpha":1.0}
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
    if not hasattr(tree, '_plot_dict'):
        

    vd.show([lns, pnt], **plot_kwargs).close()
