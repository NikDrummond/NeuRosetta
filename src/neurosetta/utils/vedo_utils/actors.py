"""Generic vedo actor helpers, independent of NeuRosetta data structures."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
from vedo import Assembly, Lines, Point, Sphere
from vedo.colors import get_color
from vedo.pointcloud.core import Points

MarkerActor = Points | Sphere

__all__ = [
    "MarkerActor",
    "combined_bounds",
    "get_marker_size",
    "make_assembly",
    "make_lines",
    "make_point_marker",
    "random_colour",
    "resolve_colour",
    "set_actor_alpha",
    "set_actor_colour",
    "set_marker_size",
]


def random_colour(seed: int | None = None) -> list[int]:
    """Return a random RGB colour as three ints in ``[0, 255]``.

    Parameters
    ----------
    seed : int | None, optional
        Seed for reproducible colours. By default None (fresh entropy).

    Returns
    -------
    list[int]
        ``[r, g, b]`` suitable for any vedo ``c=`` argument.
    """
    rng = np.random.default_rng(seed)
    return [int(v) for v in rng.integers(0, 256, size=3)]


def resolve_colour(colour: Any) -> tuple[float, float, float]:
    """Normalise any vedo colour specifier to an RGB triple in ``[0, 1]``.

    Parameters
    ----------
    colour : Any
        Colour name, hex string, index, or RGB sequence.

    Returns
    -------
    tuple[float, float, float]
        Red, green, and blue components in ``[0, 1]``.
    """
    return tuple(float(v) for v in get_color(colour))


def make_lines(starts, stops, **kwargs) -> Lines:
    """Build a vedo :class:`Lines` actor from paired start/stop coordinates.

    Parameters
    ----------
    starts, stops : array-like
        Segment endpoints with shape ``(N, 3)``.
    **kwargs
        Forwarded to the vedo ``Lines`` constructor (``c``, ``lw``, ``alpha``, ...).

    Returns
    -------
    Lines
        One cell per coordinate pair.
    """
    return Lines(np.asarray(starts), np.asarray(stops), **kwargs)


def make_point_marker(position, *, as_sphere: bool = False, **kwargs) -> MarkerActor:
    """Build a single-point marker actor.

    Parameters
    ----------
    position : array-like
        A single ``(3,)`` coordinate, or an ``(N, 3)`` array whose first row is used.
    as_sphere : bool, optional
        Return a :class:`vedo.Sphere` rather than a point. Needed for backends such
        as k3d, where single-point actors have zero average size and break export.
        By default False.
    **kwargs
        Forwarded to the marker constructor; ``r`` sets point size or sphere radius.

    Returns
    -------
    MarkerActor
        A ``Points`` actor, or a ``Sphere`` when *as_sphere* is True.
    """
    pos = np.asarray(position, dtype=float).reshape(-1, 3)[0]
    if as_sphere:
        return Sphere(pos, **kwargs)
    return Point(pos, **kwargs)


def get_marker_size(actor: MarkerActor) -> float:
    """Return the visual size of a marker actor (sphere radius or point size)."""
    if isinstance(actor, Sphere):
        return float(actor.radius)
    return float(actor.ps())


def set_marker_size(actor: MarkerActor, size: float) -> None:
    """Set the visual size of a marker actor in place.

    Spheres are scaled about their centre so the stored radius stays meaningful;
    point actors have their render size updated.
    """
    size = float(size)
    if isinstance(actor, Sphere):
        radius = float(actor.radius)
        if radius > 0:
            actor.scale(size / radius)
        actor.radius = size
    else:
        actor.ps(size)


def set_actor_colour(actor: Any, colour: Any) -> None:
    """Set an actor's colour, ignoring ``None`` actors and ``None`` colours."""
    if actor is not None and colour is not None:
        actor.color(colour)


def set_actor_alpha(actor: Any, alpha: float | None) -> None:
    """Set an actor's opacity, ignoring ``None`` actors and ``None`` values."""
    if actor is not None and alpha is not None:
        actor.alpha(float(alpha))


def combined_bounds(actors: Iterable[Any]) -> tuple[float, ...] | None:
    """Return ``(xmin, xmax, ymin, ymax, zmin, zmax)`` spanning all *actors*.

    Returns
    -------
    tuple[float, ...] | None
        None when no non-``None`` actor is supplied.
    """
    boxes = [np.asarray(a.bounds(), dtype=float) for a in actors if a is not None]
    if not boxes:
        return None
    stacked = np.vstack(boxes)
    lows = stacked[:, ::2].min(axis=0)
    highs = stacked[:, 1::2].max(axis=0)
    return tuple(float(v) for pair in zip(lows, highs, strict=True) for v in pair)


def make_assembly(actors: Iterable[Any]) -> Assembly:
    """Group actors into a single vedo :class:`Assembly`, dropping ``None`` entries."""
    return Assembly([a for a in actors if a is not None])
