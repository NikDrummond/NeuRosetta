"""Sholl profiles for a NeuRosetta tree.

Distances:

* ``radial`` — straight-line distance from a center (default the root) to a
  point on a segment of the graph that is actually passed in.
* ``path_length`` — sum of ``Path_length`` from the root along the directed path.
* ``euclidean_length`` — sum of ``Euclidean_length`` along that path.

``euclidean_length`` is not ``radial``. A path can fold back toward the root
in space while its accumulated Euclidean edge lengths keep growing.

Radial Sholl on a reduced tree uses the reduced chords. It can disagree with
the unreduced reconstruction. ``ReductionMap`` stores original edge indices
and cable lengths, not the original vertex coordinates, so the curved
geometry is not rebuilt here. ``path_length`` mode still uses the preserved
``Path_length``.

Intersections are shell crossings, not a histogram of nodes. Cumulative modes
count an edge when ``d(source) < radius <= d(target)``. Radial mode solves
the segment–sphere quadratic and ignores tangencies. A vertex on a shell is
counted on its incoming edge only. Radius 0 is not an ordinary crossing.
"""

from __future__ import annotations

from numpy import ndarray

from ...core import _Tree
from ...utils.graph_utils import (
    ShollProfile,
    sholl_area,
    sholl_peak_count,
    sholl_peak_radius,
    sholl_profile,
)
from .._doc_helpers import enrich_tree_graph_docstrings
from .tree_subtree_common import resolve_tree_edge_weights, tree_length_units


def get_sholl_profile(
    tree: _Tree,
    radii,
    mode: str = "radial",
    center: ndarray | None = None,
) -> ShollProfile:
    """Count Sholl intersections at the given radii.

    Parameters
    ----------
    tree : _Tree
        Neuron tree.
    radii : array-like
        One-dimensional, finite, strictly increasing, nonnegative radii.
        Shared across a forest: the array is not split per tree when its
        length equals the number of trees.
    mode : {"radial", "path_length", "euclidean_length"}, optional
        Distance definition. By default ``"radial"``.
    center : array-like or None, optional
        Sphere center for ``radial``. Defaults to the root coordinate.
        Not accepted for the other modes.

    Returns
    -------
    ShollProfile
        ``radii``, ``intersections``, ``mode``, ``center``, and ``units``
        when the tree defines spatial units. The graph is not modified.
        There is no ``bind`` flag: the result is a curve, not a value per node.

    Notes
    -----
    A reduced tree missing ``Path_length`` raises in ``path_length`` mode.
    Missing ``Euclidean_length`` is computed from the current endpoints,
    which on a reduced tree are the chords.
    """
    weights = None
    if mode == "path_length":
        weights = resolve_tree_edge_weights(tree, "Path_length")
    elif mode == "euclidean_length":
        weights = resolve_tree_edge_weights(tree, "Euclidean_length")
    elif mode != "radial":
        raise ValueError(
            f"Unknown Sholl mode {mode!r}. Expected 'radial', 'path_length', or 'euclidean_length'."
        )
    return sholl_profile(
        tree.graph,
        radii,
        mode=mode,
        center=center,
        weights=weights,
        units=tree_length_units(tree),
    )


def get_sholl_peak_count(profile: ShollProfile) -> int:
    """Largest intersection count on a Sholl profile.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`get_sholl_profile`.

    Returns
    -------
    int
        Maximum of ``profile.intersections``.

    See Also
    --------
    get_sholl_profile
    get_sholl_peak_radius
    get_sholl_area
    """
    return sholl_peak_count(profile)


def get_sholl_peak_radius(profile: ShollProfile) -> float:
    """Smallest radius at which the intersection count is maximal.

    If several radii share that count, the smallest radius is returned.
    Radii are strictly increasing, so this is the first maximum.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`get_sholl_profile`.

    Returns
    -------
    float
        Peak radius.

    See Also
    --------
    get_sholl_profile
    get_sholl_peak_count
    get_sholl_area
    """
    return sholl_peak_radius(profile)


def get_sholl_area(profile: ShollProfile) -> float:
    """Trapezoidal area under a Sholl curve.

    The radii do not need to be uniformly spaced. Fewer than two radii
    integrate to 0.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`get_sholl_profile`.

    Returns
    -------
    float
        Area under intersections versus radii.

    See Also
    --------
    get_sholl_profile
    get_sholl_peak_count
    get_sholl_peak_radius
    """
    return sholl_area(profile)


enrich_tree_graph_docstrings(globals())
