"""Forest-level TMD / persistence collection operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ...core import _Forest
from ...utils.topology import get_bound_tmd, pairwise_distances
from ...utils.topology.representations import persistence_images as _persistence_images
from ..tree_graphs.tree_topology import (
    persistence_diagram as tree_persistence_diagram,
)
from ..tree_graphs.tree_topology import (
    tmd as tree_tmd,
)

TopologyMetric = Literal["wasserstein", "bottleneck", "sliced_wasserstein", "barcode"]


def tmds(
    forest: _Forest,
    func: str = "Root_distance",
    bind: bool = False,
    *,
    recalculate: bool = False,
) -> list[dict[str, Any]]:
    """Compute TMDs for every tree in Forest order.

    Parameters
    ----------
    forest : _Forest
        Tree collection.
    func : str, optional
        Filtration property name. By default ``\"Root_distance\"``.
    bind : bool, optional
        Bind each TMD onto the corresponding graph. By default False.
    recalculate : bool, optional
        Recompute even if a compatible bound TMD exists. By default False.

    Returns
    -------
    list of dict
        One TMD result per tree (always returned, including when ``bind=True``).
    """
    results: list[dict[str, Any]] = []
    for tree in forest:
        result = tree_tmd(tree, func=func, bind=bind, recalculate=recalculate)
        if result is None:
            result = get_bound_tmd(tree.graph, func)
        if result is None:
            # Fallback: force a non-binding compute (should be unreachable).
            result = tree_tmd(tree, func=func, bind=False, recalculate=True)
        assert result is not None
        results.append(result)
    return results


def persistence_diagrams(
    forest: _Forest,
    func: str | None = None,
    normalize: bool = True,
    *,
    recalculate: bool = False,
) -> list[NDArray[np.float64]]:
    """Persistence diagrams for every tree, preserving Forest order."""
    return [
        tree_persistence_diagram(
            tree,
            func=func,
            normalize=normalize,
            recalculate=recalculate,
        )
        for tree in forest
    ]


def persistence_images(
    forest: _Forest,
    func: str | None = None,
    *,
    transformer: Any | None = None,
    bandwidth: float = 1.0,
    resolution: tuple[int, int] = (20, 20),
    im_range: list[float] | None = None,
    weight: Any | None = None,
    normalize: bool = True,
    recalculate: bool = False,
) -> tuple[NDArray[np.float64], Any]:
    """Fit/transform persistence images on the whole Forest.

    One transformer is fitted on all diagrams (shared coordinate range).
    Returns ``(images, transformer)`` with ``images`` shape
    ``(n_trees, resolution[0] * resolution[1])``.
    """
    diagrams = persistence_diagrams(
        forest,
        func=func,
        normalize=normalize,
        recalculate=recalculate,
    )
    return _persistence_images(
        diagrams,
        transformer=transformer,
        bandwidth=bandwidth,
        resolution=resolution,
        im_range=im_range,
        weight=weight,
    )


def topology_distance_matrix(
    forest: _Forest,
    func: str = "Root_distance",
    metric: TopologyMetric = "wasserstein",
    n_jobs: int | None = None,
    normalize: bool = True,
    *,
    recalculate: bool = False,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Pairwise persistence-diagram distance matrix in Forest order.

    Returns an ``N x N`` float array. Diagonal is zero for standard metrics.
    """
    diagrams = persistence_diagrams(
        forest,
        func=func,
        normalize=normalize,
        recalculate=recalculate,
    )
    return pairwise_distances(diagrams, metric=metric, n_jobs=n_jobs, **kwargs)
