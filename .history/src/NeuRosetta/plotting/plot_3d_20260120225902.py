import vedo as vd
from ..core import _Tree
from .utils import _build_3d
from .viewer import Viewer




def plot_3d(tree: _Tree, 
            line_kwargs:dict = {"c":"k4", "lw":1,"alpha":1.0}, 
            root_kwargs: dict = {"r":12, "c": 'k4', "alpha":1.0},
            plot_kwargs: dict = {},
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
        

    Returns
    -------
    vd.Plotter
        _description_
    """

    # get the plottable objects
    if not hasattr(tree, '_plot_dict'):
        plot_dict = _build_3d(tree = tree, line_kwargs = line_kwargs, root_kwargs = root_kwargs, cache = False)
    else:
        plot_dict = tree._plot_dict

    # if we want to cache them do so
    if cache:
        tree._plot_dict = plot_dict

    # set up viewer
    view = Viewer()
    view.add([plot_dict['lns'], plot_dict['root']])
    # show and set to close upon close
    view.show(**plot_kwargs).close()
