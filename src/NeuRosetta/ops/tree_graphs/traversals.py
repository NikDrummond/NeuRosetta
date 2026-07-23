"""Tree graph traversal functions."""
from __future__ import annotations

from typing import Iterable
from numpy import ndarray
from graph_tool.all import BFSVisitor, DFSVisitor, Graph

from ...core import _Tree

from ...utils.graph_utils import (
    bfsearch,
    bf_iterator,
    dfsearch,
    df_iterator,
    TreeDepthVisitor,
    PostOrderVisitor,
    root_index,
)


### Generic BF Search


def breadth_first_search(
    tree: _Tree,
    visitor: BFSVisitor,
    init_kwargs: dict | None = None,
    init_vertex_properties: dict | None = None,
    init_edge_properties: dict | None = None,
    root: int | None = None,
    bind: bool = True,
) -> BFSVisitor:
    """Wrapper for breadth-first search on a tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    visitor : BFSVisitor
        Visitor object that defines the behaviour of the search.
    init_kwargs : dict, optional
        Keyword arguments passed to the visitor constructor. By default None.
    init_vertex_properties : dict, optional
        Dictionary of vertex property name and type pairs to initialize before
        the search. For example, {"depth": "int"} creates a vertex property
        called "depth" of type int and passes it to the visitor. By default None.
    init_edge_properties : dict, optional
        Dictionary of edge property name and type pairs to initialize before
        the search. By default None.
    root : int | None, optional
        Index of the root vertex to start the search from. If None, uses the
        tree's root index. By default None.
    bind : bool, optional
        Whether to bind the initialized properties to the graph after the search.
        By default True.

    Returns
    -------
    BFSVisitor
        The visitor instance after the search.
    """
    if root is None:
        root = root_index(tree.graph)

    return bfsearch(
        g=tree.graph,
        visitor=visitor,
        init_kwargs=init_kwargs,
        init_vertex_properties=init_vertex_properties,
        init_edge_properties=init_edge_properties,
        root=root,
        bind=bind,
    )


def breadth_first_iterator(
    tree: _Tree, root: int | None = None, array: bool = True
) -> Iterable | ndarray:
    """Return breadth-first search iterator or array of visited edges from root.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        Root node index. If None, uses the tree's root index. By default None.
    array : bool, optional
        If False, returns iterator over edges. If True, returns numpy.ndarray.
        By default True.

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in breadth-first search traversal order.
    """
    if root is None:
        root = root_index(tree.graph)

    return bf_iterator(tree.graph, root, array)


## Generic DF Search


def depth_first_search(
    tree: _Tree,
    visitor: DFSVisitor,
    init_kwargs: dict | None = None,
    init_vertex_properties: dict | None = None,
    init_edge_properties: dict | None = None,
    root: int | None = None,
    bind: bool = True,
) -> DFSVisitor:
    """Wrapper for depth-first search on a tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    visitor : DFSVisitor
        Visitor object that defines the behaviour of the search.
    init_kwargs : dict | None, optional
        Keyword arguments passed to the visitor constructor. By default None.
    init_vertex_properties : dict | None, optional
        Dictionary of vertex property name and type pairs to initialize before
        the search. By default None.
    init_edge_properties : dict | None, optional
        Dictionary of edge property name and type pairs to initialize before
        the search. By default None.
    root : int | None, optional
        Index of the root vertex to start the search from. If None, uses the
        tree's root index. By default None.
    bind : bool, optional
        Whether to bind the initialized properties to the graph after the search.
        By default True.

    Returns
    -------
    DFSVisitor
        The visitor instance after the search.
    """
    if root is None:
        root = root_index(tree.graph)

    return dfsearch(
        g=tree.graph,
        visitor=visitor,
        init_kwargs=init_kwargs,
        init_vertex_properties=init_vertex_properties,
        init_edge_properties=init_edge_properties,
        root=root,
        bind=bind,
    )


def depth_first_iterator(
    tree: _Tree, root: int | None = None, array: bool = True
) -> Iterable | ndarray:
    """Return depth-first search iterator or array of visited edges from root.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        Root node index. If None, uses the tree's root index. By default None.
    array : bool, optional
        If False, returns iterator over edges. If True, returns numpy.ndarray.
        By default True.

    Returns
    -------
    Iterable | ndarray
        Iterator or array of edges in depth-first search traversal order.
    """
    if root is None:
        root = root_index(tree.graph)

    return df_iterator(tree.graph, root, array)


### Specific Implementations


# depths
def compute_tree_depths(tree: _Tree, root: int | None = None, bind: bool = True):
    """Compute node depths from root using breadth-first search.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        Root vertex index. If None, uses the tree's root index. By default None.
    bind : bool, optional
        If True, binds the "depth" vertex property to the graph and returns None.
        If False, returns the depth array. By default True.

    Returns
    -------
    ndarray | None
        Array of node depths if bind=False, otherwise None.
    """
    if root is None:
        root = tree.root_index()

    out = breadth_first_search(
        tree=tree,
        visitor=TreeDepthVisitor,
        init_vertex_properties={"depth": "int"},
        root=root,
        bind=bind,
    )

    return out["depth"].a if not bind else None


### Specific DF searches


# post-order traversal
def compute_post_order(tree: _Tree, root: int | None = None, bind: bool = True):
    """Get post-order traversal of tree using depth-first search.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    root : int | None, optional
        Root vertex index. If None, uses the tree's root index. By default None.
    bind : bool, optional
        If True, binds the "post_order" vertex property to the graph and returns
        None. If False, returns the PostOrderVisitor. By default True.

    Returns
    -------
    PostOrderVisitor | None
        Visitor with post_order list if bind=False, otherwise None.
    """
    if root is None:
        root = tree.root_index()

    vis = depth_first_search(tree=tree, visitor=PostOrderVisitor, root=root)

    if bind:
        tree.graph.vertex_properties["post_order"] = tree.graph.new_vp(
            "int", vis.post_order
        )
    else:
        return vis