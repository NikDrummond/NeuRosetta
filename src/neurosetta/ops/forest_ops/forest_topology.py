"""Forest-level TMD / persistence collection operations."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ...config import resolve_max_workers, resolve_parallel, resolve_show_progress
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


def _resolve_pairwise_n_jobs(
    parallel: bool | None,
    max_workers: int | None,
) -> int:
    """Map Forest parallel settings to a ``pairwise_distances`` ``n_jobs`` value."""
    if not resolve_parallel(explicit=parallel, scope="forest"):
        return 1
    mw = resolve_max_workers(explicit=max_workers)
    return -1 if mw is None else int(mw)


def tmds(
    forest: _Forest,
    func: str = "Root_distance",
    bind: bool = False,
    *,
    recalculate: bool = False,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
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
    parallel, max_workers, show_progress
        Same semantics as :meth:`~neurosetta.api.forest_class.Forest.apply`.

    Returns
    -------
    list of dict
        One TMD result per tree (always returned, including when ``bind=True``).
    """
    if bind:
        forest.apply(
            tree_tmd,
            func=func,
            bind=True,
            recalculate=recalculate,
            parallel=parallel,
            max_workers=max_workers,
            show_progress=show_progress,
        )
        out: list[dict[str, Any]] = []
        for tree in forest:
            result = get_bound_tmd(tree.graph, func)
            if result is None:
                result = tree_tmd(tree, func=func, bind=False, recalculate=True)
            assert result is not None
            out.append(result)
        return out

    results = forest.apply(
        tree_tmd,
        func=func,
        bind=False,
        recalculate=recalculate,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    assert results is not None
    return list(results)


def persistence_diagrams(
    forest: _Forest,
    func: str | None = None,
    normalize: bool = True,
    *,
    recalculate: bool = False,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
) -> list[NDArray[np.float64]]:
    """Persistence diagrams for every tree, preserving Forest order.

    Parameters
    ----------
    parallel, max_workers, show_progress
        Same semantics as :meth:`~neurosetta.api.forest_class.Forest.apply`.
    """
    results = forest.apply(
        tree_persistence_diagram,
        func=func,
        normalize=normalize,
        recalculate=recalculate,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    assert results is not None
    return list(results)


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
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
) -> tuple[NDArray[np.float64], Any]:
    """Fit/transform persistence images on the whole Forest.

    One transformer is fitted on all diagrams (shared coordinate range).
    Returns ``(images, transformer)`` with ``images`` shape
    ``(n_trees, resolution[0] * resolution[1])``.

    Diagram extraction uses ``parallel`` / ``max_workers`` / ``show_progress``
    like :meth:`~neurosetta.api.forest_class.Forest.apply`.
    """
    diagrams = persistence_diagrams(
        forest,
        func=func,
        normalize=normalize,
        recalculate=recalculate,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
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
    normalize: bool = True,
    *,
    recalculate: bool = False,
    parallel: bool | None = None,
    max_workers: int | None = None,
    show_progress: bool | None = None,
    **kwargs: Any,
) -> NDArray[np.float64]:
    """Pairwise persistence-diagram distance matrix in Forest order.

    Two sequential stages (no nested pools):

    1. Persistence diagrams per tree (``Forest.apply`` parallelism).
    2. Upper-triangle pairwise distances (``n_jobs`` derived from ``parallel`` /
       ``max_workers``).

    Parameters
    ----------
    forest : _Forest
        Tree collection.
    func : str, optional
        Filtration property. By default ``\"Root_distance\"``.
    metric : {\"wasserstein\", \"bottleneck\", \"sliced_wasserstein\", \"barcode\"}
        Pairwise metric.
    normalize : bool, optional
        Normalize TMD endpoints before distance. By default True.
    recalculate : bool, optional
        Recompute TMDs even if bound. By default False.
    parallel : bool or None, optional
        Enable parallelism for both stages. When None, use the global
        forest-scoped setting. By default None.
    max_workers : int or None, optional
        Worker limit. When None and parallel, use all CPUs for pairwise
        (``n_jobs=-1``). By default None.
    show_progress : bool or None, optional
        Show tqdm progress bars for diagram extraction and pairwise pairs.
        When None, use the global setting. By default None.
    **kwargs
        Forwarded to the low-level distance implementation.

    Returns
    -------
    ndarray
        ``N x N`` float array. Diagonal is zero for standard metrics.
    """
    show = resolve_show_progress(explicit=show_progress)
    diagrams = persistence_diagrams(
        forest,
        func=func,
        normalize=normalize,
        recalculate=recalculate,
        parallel=parallel,
        max_workers=max_workers,
        show_progress=show_progress,
    )
    n_jobs = _resolve_pairwise_n_jobs(parallel, max_workers)
    return pairwise_distances(
        diagrams,
        metric=metric,
        n_jobs=n_jobs,
        show_progress=show,
        **kwargs,
    )
