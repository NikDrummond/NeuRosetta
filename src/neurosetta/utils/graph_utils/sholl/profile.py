"""Sholl profiles and scalar summaries of a profile.

The profile is a function of radius, not a per-vertex map, so nothing is
bound onto the graph. Three distance definitions are supported:

``radial``
    Straight-line distance from a center (default: the root coordinate) to
    a point on a straight segment of the input graph. This is an embedding
    measurement. On a reduced tree the segments are the stored chords, not
    the original curved cable. :class:`ReductionMap` records which original
    edges were collapsed and their cable lengths; it does not store the
    original vertex coordinates, so those curves are not reconstructed here.

``path_length``
    Sum of ``Path_length`` along the directed path from the root. This is
    physical cable. It is not recomputed from coordinates.

``euclidean_length``
    Sum of ``Euclidean_length`` along that same path. This is **not** radial
    distance: each edge contributes its own straight-line length, and the
    path can fold back toward the root in space.
"""

from __future__ import annotations

from dataclasses import dataclass

from graph_tool.all import Graph
from numpy import asarray, float64, int64, isfinite, ndarray

try:
    from numpy import trapezoid
except ImportError:  # NumPy < 2
    from numpy import trapz as trapezoid

from ..subtrees._accumulate import prepare_subtree_traversal
from ..tree_validation import validate_rooted_tree
from .distances import cumulative_root_distance
from .intersections import (
    count_distance_crossings,
    count_radial_intersections,
    edge_list,
    validate_radii,
    vertex_coordinates,
)

_MODE_PROPERTY = {
    "path_length": "Path_length",
    "euclidean_length": "Euclidean_length",
}


@dataclass(frozen=True, slots=True)
class ShollProfile:
    """Intersections of a morphology with a sequence of distance shells.

    Attributes
    ----------
    radii : ndarray
        Radii at which intersections were counted. ``float64``, shape ``(R,)``.
    intersections : ndarray
        Intersection counts, ``int64``, shape ``(R,)``.
    mode : str
        ``"radial"``, ``"path_length"``, or ``"euclidean_length"``.
    center : ndarray or None
        Sphere center for ``radial`` mode, shape ``(3,)``. ``None`` for
        cumulative path modes, which are measured from the root along edges.
    units : str or None
        Optional unit name supplied by a caller. This package does not
        interpret Pint quantities; the graph-level function leaves this
        ``None``.
    """

    radii: ndarray
    intersections: ndarray
    mode: str
    center: ndarray | None = None
    units: str | None = None

    def __post_init__(self) -> None:
        radii = asarray(self.radii, dtype=float64).copy()
        intersections = asarray(self.intersections, dtype=int64).copy()
        if radii.shape != intersections.shape or radii.ndim != 1:
            raise ValueError("radii and intersections must be 1-d arrays of equal length.")
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "intersections", intersections)
        if self.center is None:
            return
        center = asarray(self.center, dtype=float64).reshape(3).copy()
        object.__setattr__(self, "center", center)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShollProfile):
            return NotImplemented
        if self.mode != other.mode or self.units != other.units:
            return False
        if not (self.radii.shape == other.radii.shape and (self.radii == other.radii).all()):
            return False
        if not (
            self.intersections.shape == other.intersections.shape
            and (self.intersections == other.intersections).all()
        ):
            return False
        if self.center is None or other.center is None:
            return self.center is None and other.center is None
        return bool((self.center == other.center).all())


def _as_center(center) -> ndarray:
    arr = asarray(center, dtype=float64)
    if arr.shape != (3,):
        raise ValueError(f"center must have shape (3,), got {arr.shape}.")
    if not isfinite(arr).all():
        raise ValueError("center must be finite.")
    return arr


def sholl_profile(
    g: Graph,
    radii,
    mode: str = "radial",
    center: ndarray | None = None,
    *,
    weights: ndarray | None = None,
    units: str | None = None,
) -> ShollProfile:
    """Count shell intersections along a directed rooted tree.

    Parameters
    ----------
    g : Graph
        Directed rooted tree. Radial mode needs vertex properties ``x``,
        ``y``, and ``z``. ``path_length`` needs edge property ``Path_length``
        unless *weights* is passed. ``euclidean_length`` needs
        ``Euclidean_length`` unless *weights* is passed.
    radii : array-like
        One-dimensional, finite, strictly increasing, nonnegative radii.
    mode : {"radial", "path_length", "euclidean_length"}, optional
        Distance definition. By default ``"radial"``.
    center : array-like or None, optional
        Sphere center for ``radial`` mode. Defaults to the root coordinate.
        Rejected for the cumulative modes: those distances start at the root
        and follow edges.
    weights : ndarray or None, optional
        Explicit nonnegative edge weights in edge-index order, used by the
        cumulative modes instead of the named property. Ignored for
        ``radial``, which does not use edge length.
    units : str or None, optional
        Stored on the result and otherwise unused. By default None.

    Returns
    -------
    ShollProfile
        Radii, intersection counts, mode, center, and units.

    Raises
    ------
    ValueError
        If the graph, mode, radii, center, or required properties are invalid.

    Notes
    -----
    The graph is not modified. In particular, root-distance maps are not
    written to ``Root_distance``.

    Radial Sholl on a reduced graph uses the straight chords of the reduced
    edges. That profile can disagree with radial Sholl of the unreduced
    reconstruction even when ``Path_length`` was preserved. Path-length Sholl
    still uses the preserved cable.

    See Also
    --------
    sholl_peak_count
    sholl_peak_radius
    sholl_area
    """
    if mode not in _MODE_PROPERTY and mode != "radial":
        raise ValueError(
            f"Unknown Sholl mode {mode!r}. Expected 'radial', 'path_length', or 'euclidean_length'."
        )
    if center is not None and mode != "radial":
        raise ValueError(
            "An explicit center is only supported for mode='radial'. "
            "path_length and euclidean_length are measured from the root along edges."
        )
    checked = validate_radii(radii)
    root = validate_rooted_tree(g)
    edges = edge_list(g)

    if mode == "radial":
        coords = vertex_coordinates(g)
        origin = coords[root] if center is None else _as_center(center)
        counts = count_radial_intersections(coords, edges, checked, origin)
        return ShollProfile(
            radii=checked,
            intersections=counts,
            mode=mode,
            center=origin,
            units=units,
        )

    traversal = prepare_subtree_traversal(g)
    distance = cumulative_root_distance(g, traversal, _MODE_PROPERTY[mode], weights)
    counts = count_distance_crossings(distance, edges, checked)
    return ShollProfile(
        radii=checked,
        intersections=counts,
        mode=mode,
        center=None,
        units=units,
    )


def sholl_peak_count(profile: ShollProfile) -> int:
    """Return the largest intersection count on *profile*.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`sholl_profile`.

    Returns
    -------
    int
        ``max(intersections)``, or 0 when *profile* has no radii.
    """
    if profile.intersections.size == 0:
        return 0
    return int(profile.intersections.max())


def sholl_peak_radius(profile: ShollProfile) -> float:
    """Return the smallest radius that attains the peak intersection count.

    Ties take the first maximum. Radii are strictly increasing, so that is
    the smallest radius among the tied shells.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`sholl_profile`.

    Returns
    -------
    float
        Peak radius, or NaN when *profile* has no radii.
    """
    if profile.intersections.size == 0:
        return float("nan")
    peak = int(profile.intersections.max())
    return float(profile.radii[profile.intersections == peak][0])


def sholl_area(profile: ShollProfile) -> float:
    """Integrate the Sholl curve with the trapezoidal rule.

    Radii need not be uniform. A profile with fewer than two radii has area 0
    because there is no interval to integrate.

    Parameters
    ----------
    profile : ShollProfile
        Result of :func:`sholl_profile`.

    Returns
    -------
    float
        Area under ``intersections`` versus ``radii``.
    """
    if profile.radii.size < 2:
        return 0.0
    return float(trapezoid(profile.intersections.astype(float64), profile.radii))
