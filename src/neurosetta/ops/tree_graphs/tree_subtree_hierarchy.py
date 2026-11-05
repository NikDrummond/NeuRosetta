"""Strahler order on a NeuRosetta tree.

Leaves have order 1. One child: inherit that child's order. Several children:
let ``m`` be the largest child order. If at least two children have order
``m``, the node is ``m + 1``; otherwise it is ``m``. Multifurcations use that
same rule.

Strahler order is a topological label. It does not use coordinates or edge
lengths, and it is not the number of branches between the root and the node.
Chains of transitive nodes inherit the downstream order, so inserting them
does not change the order at the original nodes.
"""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import strahler_order
from .._doc_helpers import enrich_tree_graph_docstrings


def get_strahler_order(tree: _Tree, bind: bool = False) -> ndarray | None:
    """Compute Strahler order at every node.

    Parameters
    ----------
    tree : _Tree
        Neuron tree, including a single node and a root that is not index 0.
    bind : bool, optional
        If True, store vertex property ``strahler_order`` and return None.
        Other properties are not removed. By default False.

    Returns
    -------
    ndarray or None
        Int64 order per node when ``bind`` is False. Leaves are 1.
    """
    return strahler_order(tree.graph, bind=bind)


def get_max_strahler_order(tree: _Tree) -> int:
    """Return the largest Strahler order in the tree.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.

    Returns
    -------
    int
        Maximum of :func:`get_strahler_order`. The graph is not modified.
    """
    orders = get_strahler_order(tree, bind=False)
    if orders is None:
        raise RuntimeError("get_strahler_order(bind=False) did not return an array.")
    return int(orders.max())


enrich_tree_graph_docstrings(globals())
