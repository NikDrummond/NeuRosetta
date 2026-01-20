### 2D plotting of neurons

from numpy import array, stack, vstack, ones, ndarray
from matplotlib.collections import LineCollection

from matplotlib.pyplot import Axes, subplots

from ..core import _Tree

def plot_2d(tree: _Tree, 
            center:ndarray = array([0,0]), 
            ax_pad:float = 1, 
            line_kwargs:dict = {"color":'gray',"linewidth":1,'alpha':1},
            point_kwargs:dict = {"color":'k'},
            root_kwargs:dict = {"color":'r'},
            axes:Axes | None = None):

    """Generates a simple 2D plot of a neuron morphology

    Parameters
    ----------
    tree : Tree_graph
        Neuron tree
    center : ndarray
        An array of the 

    """

    ### generate lines

    # get source and target coordinates
    starts, stops = tree.get_edge_coordinates()
    # remove z axis
    starts = starts[:,[0,1]]
    stops = stops[:,[0,1]]

    starts += center
    stops += center
    # stack starts and end into a segments array
    segments = stack([starts, stops], axis = 1)
    # create LineCollection
    lc = LineCollection(segments, **line_kwargs)

    if axes is None:
        fig, axes = subplots()

    axes.add_collection(lc)

    # plot points
    coords = tree.get_node_coordinates()

    # generate mask to subset out root
    mask = ones(coords.shape[0], dtype = bool)
    mask[tree.root_index()] = False


    axes.scatter(coords[mask,0], coords[mask,1], **point_kwargs)
    axes.scatter(coords[~mask,0], coords[~mask,1], **root_kwargs)

    # adjust limits
    all_pts = vstack((starts, stops))
    axes.set_xlim(all_pts[:,0].min() - ax_pad, all_pts[:,0].max() + ax_pad)
    axes.set_ylim(all_pts[:,1].min() - ax_pad, all_pts[:,1].max() + ax_pad)

    axes.set_aspect('equal')
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)