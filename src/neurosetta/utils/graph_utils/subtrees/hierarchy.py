"""Strahler order on a directed rooted tree.

Leaves have order 1. A vertex with one child inherits that child's order.
With several children, let ``m`` be the maximum child order. If at least two
children have order ``m``, the vertex's order is ``m + 1``. Otherwise it
keeps ``m``. Multifurcations use the same rule as binary nodes.

Strahler order depends only on the branching hierarchy. Coordinates, edge
lengths, and the root index do not enter the calculation. It is not the
number of branches between the root and the vertex, and it is not
:func:`subgraph_branch_depth`.

Inserting transitive (degree-2) vertices does not change the order of the
original vertices: a chain inherits the downstream order unchanged.
"""

from __future__ import annotations

from graph_tool.all import Graph
from numpy import int64, ndarray, ones

from ._accumulate import bind_or_return, prepare_subtree_traversal


def strahler_order(g: Graph, bind: bool = True) -> ndarray | None:
    """Compute Strahler order for every vertex.

    Parameters
    ----------
    g : Graph
        Directed rooted tree, including a single vertex and trees whose root
        is not vertex 0. Active filters are rejected.
    bind : bool, optional
        If True, store vertex property ``strahler_order`` and return None.
        Other properties are left alone. If False, return an ``int64`` array
        and do not modify *g*. By default True.

    Returns
    -------
    ndarray or None
        One Strahler order per vertex when ``bind`` is False. Leaves are 1.

    Notes
    -----
    The walk is reverse topological and runs in O(V + E). It does not recurse.

    See Also
    --------
    max_strahler_order
    subgraph_branch_depth
    """
    traversal = prepare_subtree_traversal(g)
    order = ones(len(traversal.children), dtype=int64)
    for vertex in traversal.order:
        kids = traversal.children[int(vertex)]
        if not kids:
            order[int(vertex)] = 1
            continue
        child_orders = order[kids]
        maximum = int(child_orders.max())
        n_at_max = int((child_orders == maximum).sum())
        order[int(vertex)] = maximum + 1 if n_at_max >= 2 else maximum
    return bind_or_return(g, "strahler_order", order, "int64_t", bind)


def max_strahler_order(g: Graph) -> int:
    """Return the largest Strahler order in *g*.

    Parameters
    ----------
    g : Graph
        Directed rooted tree.

    Returns
    -------
    int
        Maximum of :func:`strahler_order`. The graph is not modified.
    """
    orders = strahler_order(g, bind=False)
    if orders is None:
        raise RuntimeError("strahler_order(bind=False) did not return an array.")
    return int(orders.max())
