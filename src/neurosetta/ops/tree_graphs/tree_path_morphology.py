"""Branch and section counts along one path."""

from __future__ import annotations

from ...core import _Tree
from ...utils.graph_utils.paths import path_branch_count, path_section_count
from .._doc_helpers import enrich_tree_graph_docstrings


def get_path_branch_count(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
    include_endpoints: bool = False,
) -> int | None:
    """Count branch nodes on the path.

    A branch node has at least two children. Endpoints are excluded unless
    requested.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.
    include_endpoints : bool, optional
        Count endpoints that are branch nodes. By default False.

    Returns
    -------
    int or None
        Count, or ``None`` when the path does not exist.
    """
    return path_branch_count(
        tree.graph,
        source,
        target,
        directed=directed,
        include_endpoints=include_endpoints,
    )


def get_path_section_count(
    tree: _Tree,
    source: int,
    target: int,
    *,
    directed: bool = True,
) -> int | None:
    """Count morphological sections travelled by the path.

    A section runs between consecutive landmarks: the root, a branch node,
    or a leaf. Transitive nodes are inside a section. A path that starts
    or ends inside a section still counts that partial section. A path
    from a node to itself counts 0.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    source, target : int
        Path endpoints.
    directed : bool, optional
        Stored direction when True. By default True.

    Returns
    -------
    int or None
        Section count, or ``None`` when the path does not exist.
    """
    return path_section_count(tree.graph, source, target, directed=directed)


enrich_tree_graph_docstrings(globals())
