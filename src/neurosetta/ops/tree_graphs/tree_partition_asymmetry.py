"""Partition asymmetry on a NeuRosetta tree.

Equations, defaults, multifurcation averaging, ``include_root``, and property
names match the historical implementation. Cable modes require ``Path_length``
and include each daughter's connecting edge. Weighted mode multiplies the
unweighted leaf asymmetry by the fraction of total cable downstream of the
branch. Non-branch nodes are 0.
"""

from __future__ import annotations

from typing import Literal

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import partition_asymmetry
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_checks import has_property


def get_partition_asymmetry(
    tree: _Tree,
    mode: Literal["unweighted", "weighted", "van_pelt", "cable"] = "weighted",
    bind: bool = False,
    *,
    include_root: bool = True,
    property_name: str | None = None,
) -> ndarray | None:
    r"""Compute partition asymmetry for a NeuRosetta Tree.

    Supported modes
    ---------------
    * unweighted — ``abs(r - s) / (r + s - 1)``
    * weighted — unweighted asymmetry times the fraction of total
      cable in the downstream subtree
    * van_pelt — ``abs(r - s) / (r + s - 2)``; the (1, 1) partition is 0
    * cable — ``abs(C1 - C2) / (C1 + C2)``, with C1 and C2 the daughter
      subtree cable lengths, including their connecting edges

    For multifurcations, scores are averaged over all unordered
    pairs of daughter subtrees.

    Parameters
    ----------
    tree : _Tree
        NeuRosetta Tree object.
    mode : str, default "weighted"
        Partition asymmetry definition.
    bind : bool, default False
        If True, attach results as a vertex property and return None.
        Otherwise, return an ndarray.
    include_root : bool, default True
        Whether to include the root when it is a branching node.
    property_name : str or None, default None
        Vertex property name used when bind=True.
        Defaults to ``partition_asymmetry_<mode>``.

    Returns
    -------
    ndarray | None
        One asymmetry score per vertex, or None when bind=True.

    Notes
    -----
    Cable-based modes require the "Path_length" edge property
    containing true dendritic cable lengths.

    Euclidean distances must not be substituted for true cable
    lengths on reduced trees.
    """
    valid_modes = {"unweighted", "weighted", "van_pelt", "cable"}

    if mode not in valid_modes:
        raise ValueError(
            f"Unknown partition asymmetry mode: {mode!r}. Expected one of {sorted(valid_modes)}."
        )

    if mode in {"weighted", "cable"} and not has_property(tree, "Path_length", "e"):
        raise ValueError(
            f"Mode {mode!r} requires the 'Path_length' "
            "edge property containing true dendritic cable lengths."
        )

    return partition_asymmetry(
        tree.graph,
        mode=mode,
        bind=bind,
        include_root=include_root,
        property_name=property_name,
    )


enrich_tree_graph_docstrings(globals())
