"""Distance helpers for mesh Stones."""

from __future__ import annotations

from numpy import ndarray

from ...core import _Mesh
from ...utils.vedo_utils.surface_distances import surface_distance


def distance_to_surface(
    mesh: _Mesh,
    points: ndarray,
    *,
    method: str = "surface",
    face_indices=None,
    triangle_method: str = "auto",
    k_candidates: int = 64,
) -> ndarray:
    """Distances from *points* to this mesh's surface.

    Thin wrapper around
    :func:`~neurosetta.utils.vedo_utils.surface_distances.surface_distance`
    that returns distances only.

    Parameters
    ----------
    mesh : _Mesh
        Neuron or neuropil mesh.
    points : ndarray
        Query points, shape ``(N, 3)``.
    method, face_indices, triangle_method, k_candidates
        Forwarded to :func:`~neurosetta.utils.vedo_utils.surface_distances.surface_distance`.

    Returns
    -------
    ndarray
        Distances, shape ``(N,)``.
    """
    dists, _ = surface_distance(
        points,
        mesh.mesh,
        face_indices=face_indices,
        method=method,  # type: ignore[arg-type]
        triangle_method=triangle_method,
        k_candidates=k_candidates,
    )
    return dists
