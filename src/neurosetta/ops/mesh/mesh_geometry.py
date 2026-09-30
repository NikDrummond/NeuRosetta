"""Lightweight geometry queries for mesh Stones."""

from __future__ import annotations

import numpy as np
from numpy import ndarray

from ...core import _Mesh


def count_vertices(mesh: _Mesh) -> int:
    """Return the number of mesh vertices."""
    return int(mesh.mesh.npoints)


def count_faces(mesh: _Mesh) -> int:
    """Return the number of mesh faces (cells)."""
    return int(len(mesh.mesh.cells))


def get_vertex_coordinates(mesh: _Mesh) -> ndarray:
    """Return vertex coordinates as an ``(N, 3)`` float array."""
    return np.asarray(mesh.mesh.vertices, dtype=float)


def get_bounds(mesh: _Mesh) -> ndarray:
    """Return axis-aligned bounds ``(xmin, xmax, ymin, ymax, zmin, zmax)``."""
    return np.asarray(mesh.mesh.bounds(), dtype=float)
