"""Sholl analysis on directed rooted trees.

Three distance modes:

* ``radial`` — Euclidean distance from a center to a point on a segment
* ``path_length`` — accumulated ``Path_length`` from the root
* ``euclidean_length`` — accumulated ``Euclidean_length`` from the root

``euclidean_length`` is a path sum. It is not ``radial``.
"""

from __future__ import annotations

from .profile import (
    ShollProfile,
    sholl_area,
    sholl_peak_count,
    sholl_peak_radius,
    sholl_profile,
)

__all__ = [
    "ShollProfile",
    "sholl_area",
    "sholl_peak_count",
    "sholl_peak_radius",
    "sholl_profile",
]
