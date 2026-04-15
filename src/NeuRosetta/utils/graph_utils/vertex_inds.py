""" Graph functions to get indicies of vertices"""
from numpy import where, ndarray, unique, concatenate
from graph_tool.all import Graph

from .traversals import bf_iterator, df_iterator

def root_index(g: Graph) -> int:
    """Return index (int) of root node (in-degree == 0)"""
    return int(where(g.degree_property_map("in").a ==0)[0][0])

def leaf_indices(g: Graph) -> ndarray:
    """Return array of indicies of leaves (out-degree == 0)"""
    return where(g.degree_property_map("out").a == 0)[0]

def branch_indices(g: Graph) -> ndarray:
    """Return array of indices of branches (out-degree > 1)"""
    return where(g.degree_property_map("out").a > 1)[0]

def core_indices(g: Graph, include_root: bool = True) -> ndarray:
    """Return array of indicies of cvore vertices (branches and leaves). Include root optionally"""
    l_inds = leaf_indices(g)
    b_inds = branch_indices(g)

    inds = unique(concatenate([l_inds, b_inds]))

    root = root_index(g)

    root_in = root in inds

    if include_root and not root_in:
        inds = concatenate(([root],inds))
    elif not include_root and root_in:
        inds = inds[inds != root]
    return inds

def edge_indices(g: Graph, root: int | None = None, traversal_order: str = 'Breadth') -> ndarray:
    """Return Edge index array. If root is given, get the sub-tree edges from root"""
    if root is None:
        edges = g.get_edges()
    else:
        if traversal_order == "Breadth":
            edges = bf_iterator(g, root, array = True)
        elif traversal_order == "Depth":
            edges = df_iterator(g, root, array = True)
        else:
            raise ValueError(f"traversal_order must be Breadth or Depth, not {traversal_order}")
    return edges

def subtree_indices(g: Graph, root: int, traversal_order: str = "Breadth") -> ndarray:
    """Return vertex indicies of subtree from specified root"""
    if traversal_order == "Breadth":
        return unique(bf_iterator(g, root, array = True))
    if traversal_order == "Depth":
        return unique(df_iterator(g, root, array = True))
    raise ValueError(f"traversal_order must be Breadth or Depth, not {traversal_order}")
