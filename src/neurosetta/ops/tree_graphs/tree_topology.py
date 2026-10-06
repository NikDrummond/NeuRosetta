"""Tree-level topological morphology (TMD) and persistence wrappers."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from ...core import _Tree
from ...utils.graph_utils import g_has_property
from ...utils.graph_utils.gt_properties import bind_graph_property
from ...utils.topology import (
    TMD_GP,
    barcode_distance,
    bottleneck_distance,
    get_bound_tmd,
    normalize_tmd_payload,
    wasserstein_distance,
)
from ...utils.topology import persistence_diagram as _persistence_diagram
from ...utils.topology import plot_barcode as _plot_barcode
from ...utils.topology import plot_persistence_diagram as _plot_persistence_diagram
from ...utils.topology import tmd as _tmd
from ...utils.topology.representations import persistence_images, reshape_persistence_image
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_path_lengths import get_root_distance

TopologyMetric = Literal["wasserstein", "bottleneck", "sliced_wasserstein", "barcode"]


def _ensure_filtration(tree: _Tree, func: str) -> None:
    """Ensure a named filtration vertex property exists (Root_distance helper)."""
    if func == "Root_distance" and not g_has_property(tree.graph, "Root_distance", "v"):
        get_root_distance(tree, bind=True)


def tmd(
    tree: _Tree,
    func: str = "Root_distance",
    bind: bool = True,
    *,
    recalculate: bool = False,
) -> dict[str, Any] | None:
    """Compute the Topological Morphology Descriptor (Kanari et al., 2018).

    Runs on the original graph (no reduction). Persistence pairs keep original
    vertex indices. See :func:`neurosetta.utils.topology.tmd.tmd`.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    func : str, optional
        Vertex filtration property name. ``\"Root_distance\"`` is computed and
        bound if missing. By default ``\"Root_distance\"``.
    bind : bool, optional
        If True, store the TMD as graph property ``\"TMD\"`` and return None.
        The bound object is pickled into ``.nr`` with the graph. If False,
        return the result dict. By default True.
    recalculate : bool, optional
        If False and a compatible bound TMD exists, reuse it. By default False.

    Returns
    -------
    dict | None
        TMD result when ``bind=False``, otherwise None.
    """
    _ensure_filtration(tree, func)

    if not recalculate:
        cached = get_bound_tmd(tree.graph, func)
        if cached is not None:
            return None if bind else cached

    result = _tmd(tree.graph, func=func, bind=bind)
    return None if bind else result


def freeze_tmd_for_save(tree: _Tree) -> None:
    """Normalize bound ``TMD`` to a pickle-safe payload before ``.nr`` save."""
    if not g_has_property(tree.graph, TMD_GP, "g"):
        return
    val = tree.graph.gp[TMD_GP]
    if not isinstance(val, dict):
        return
    bind_graph_property(tree.graph, TMD_GP, "object", normalize_tmd_payload(val))


def get_tmd_survival_lengths(
    tree: _Tree,
    func: str = "Root_distance",
    *,
    recalculate: bool = False,
) -> NDArray[np.float64]:
    """Return TMD bar survival lengths ``|death - birth|`` (one per pair).

    Default filtration is ``Root_distance`` (path distance from root). Lengths
    are in filtration units.
    """
    result = _resolve_tmd(tree, func, recalculate=recalculate)
    return np.asarray(result["survival_len"], dtype=np.float64)


def count_tmd_bars(
    tree: _Tree,
    func: str = "Root_distance",
    *,
    recalculate: bool = False,
) -> int:
    """Return the number of TMD persistence bars (persistence pairs)."""
    return int(get_tmd_survival_lengths(tree, func=func, recalculate=recalculate).size)


def _resolve_tmd(
    tree: _Tree,
    func: str | None,
    *,
    recalculate: bool = False,
) -> dict[str, Any]:
    filt = "Root_distance" if func is None else func
    _ensure_filtration(tree, filt)

    if not recalculate:
        cached = get_bound_tmd(tree.graph, filt if func is not None else None)
        if cached is not None and (func is None or cached.get("function") == filt):
            return cached

    return _tmd(tree.graph, func=filt, bind=False)


def persistence_diagram(
    tree: _Tree,
    func: str | None = None,
    normalize: bool = True,
    *,
    recalculate: bool = False,
) -> NDArray[np.float64]:
    """Return the persistence diagram ``(N, 2)`` for a tree's TMD.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    func : str or None, optional
        Filtration property. If None, reuse a bound TMD when present, else
        compute with ``\"Root_distance\"``. By default None.
    normalize : bool, optional
        Order endpoints as ``(low, high)`` for generic TDA tools.
        By default True.
    recalculate : bool, optional
        Recompute TMD even if a bound result exists. By default False.
    """
    tmd_result = _resolve_tmd(tree, func, recalculate=recalculate)
    return _persistence_diagram(tmd_result, normalize=normalize)


def plot_persistence_diagram(
    tree: _Tree,
    func: str | None = None,
    normalize: bool = True,
    axes: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot the tree's persistence diagram (GUDHI); return Axes."""
    diagram = persistence_diagram(tree, func=func, normalize=normalize)
    return _plot_persistence_diagram(diagram, axes=axes, **kwargs)


def plot_barcode(
    tree: _Tree,
    func: str | None = None,
    normalize: bool = True,
    axes: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Plot the tree's persistence barcode (GUDHI); return Axes."""
    diagram = persistence_diagram(tree, func=func, normalize=normalize)
    return _plot_barcode(diagram, axes=axes, **kwargs)


def persistence_image(
    tree: _Tree,
    func: str | None = None,
    *,
    transformer: Any | None = None,
    bandwidth: float = 1.0,
    resolution: tuple[int, int] = (20, 20),
    im_range: list[float] | None = None,
    weight: Any | None = None,
    reshape: bool = False,
    normalize: bool = True,
) -> tuple[NDArray[np.float64], Any]:
    """Compute a persistence image for one tree.

    Prefer fitting ``transformer`` on a Forest / collection so the image
    coordinate range is shared. Returns ``(image, transformer)``.
    """
    diagram = persistence_diagram(tree, func=func, normalize=normalize)
    images, fitted = persistence_images(
        [diagram],
        transformer=transformer,
        bandwidth=bandwidth,
        resolution=resolution,
        im_range=im_range,
        weight=weight,
    )
    image = images[0]
    if reshape:
        image = reshape_persistence_image(image, resolution=resolution)
    return image, fitted


def topology_distance(
    tree: _Tree,
    other: _Tree,
    func: str = "Root_distance",
    metric: TopologyMetric = "wasserstein",
    normalize: bool = True,
    **kwargs: Any,
) -> float:
    """Distance between two trees' persistence diagrams.

    Parameters
    ----------
    tree, other : _Tree
        Trees to compare.
    func : str, optional
        Filtration property used for both trees. By default ``\"Root_distance\"``.
    metric : {\"wasserstein\", \"bottleneck\", \"sliced_wasserstein\", \"barcode\"}
        Distance. ``\"barcode\"`` is the Kanari density distance (no GUDHI).
    normalize : bool, optional
        Normalize TMD endpoints before distance. By default True.
    **kwargs
        Forwarded to the distance implementation.
    """
    diag_a = persistence_diagram(tree, func=func, normalize=normalize)
    diag_b = persistence_diagram(other, func=func, normalize=normalize)

    name = str(metric).lower()
    if name == "barcode":
        return barcode_distance(diag_a, diag_b)
    if name == "bottleneck":
        return bottleneck_distance(diag_a, diag_b, **kwargs)
    if name == "wasserstein":
        return wasserstein_distance(diag_a, diag_b, **kwargs)
    if name == "sliced_wasserstein":
        from ...utils.topology.distances import pairwise_distances

        mat = pairwise_distances(
            [diag_a, diag_b],
            metric="sliced_wasserstein",
            **kwargs,
        )
        return float(mat[0, 1])
    raise ValueError(
        f"Unknown metric {metric!r}; expected wasserstein, bottleneck, "
        "sliced_wasserstein, or barcode"
    )


enrich_tree_graph_docstrings(globals())
