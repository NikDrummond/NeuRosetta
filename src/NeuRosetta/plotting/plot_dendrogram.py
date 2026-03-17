# import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy import stack, vstack, arange
from matplotlib.pyplot import Axes, subplots

from ..core import _Tree

### compute x coordinate
def compute_dend_x(
    tree: _Tree,
    root: int,
    depth,            # vertex PropertyMap<int>
    post_order,       # PostOrderVisitor (has .post_order list)
    x_spacing: float = 1.0,
) -> dict[int, float]:
    """
    Derive x coordinates from depth and the post-order visit list.
    Children of v are simply its neighbours at depth depth[v] + 1.
    """
    x = {}
    counter = 0

    for v in post_order.post_order:
        # children = neighbours one level deeper
        ch = tree.graph.get_out_neighbors(v)
        # v is a leaf
        if not ch.size > 0:                          
            x[v] = counter * x_spacing
            counter += 1
        # v aint no leaf
        else:                               
            x[v] = sum([x[c] for c in ch]) / ch.size

    return x



def plot_dendrogram(tree: _Tree, ax_pad = 1, root_possition = 'bottom', x_spacing = 0.3, axes: Axes | None = None):

    depth = tree.Get_node_depths(bind = False)
    post_order = tree.Get_post_order_traversal(bind = False)

    x = compute_dend_x(tree, 0, depth, post_order, x_spacing = x_spacing)

    # build a drawable position property
    pos = tree.graph.new_vertex_property("vector<double>")
    for v in tree.graph.vertices():
        pos[v] = [x[int(v)], depth[v]] 

    coords = pos.get_2d_array().T
    edges = tree.edge_indices()

    starts = coords[edges[:, 0]]
    stops = coords[edges[:, 1]]

    segments = stack([starts, stops], axis=1)

    if axes is None:
        fig, axes = subplots()

    # sort out y inversion
    if root_possition == 'bottom':
        pass
    elif root_possition == 'top':
        axes.yaxis.set_inverted(True)
    else:
        raise ValueError(f"root_position must be 'top' or 'bottom', got {root_possition}")

    lc = LineCollection(segments, color = 'gray', linewidth = 1, alpha = 1)

    axes.add_collection(lc)

    # adjust limits
    all_pts = vstack((starts, stops))
    axes.set_xlim(all_pts[:, 0].min() - ax_pad, all_pts[:, 0].max() + ax_pad)
    axes.set_ylim(all_pts[:, 1].min() - ax_pad, all_pts[:, 1].max() + ax_pad)

    axes.set_aspect("equal")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    axes.spines["bottom"].set_visible(False)
    axes.set_xticks([])

    axes.scatter(coords[0,0],coords[0,1], c = 'r', zorder = 100)

    axes.set_ylabel("Tree Depth")
    axes.set_aspect("equal")
    axes.set_yticks(arange(0,depth.a.max() + 1,2));
