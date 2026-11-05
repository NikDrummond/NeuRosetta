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
    method : str, optional
        Distance method forwarded to
        :func:`~neurosetta.utils.vedo_utils.surface_distances.surface_distance`.
        By default ``\"surface\"``.
    face_indices : array-like or None, optional
        Restrict the query to these faces. By default None.
    triangle_method : str, optional
        Triangle-query strategy. By default ``\"auto\"``.
    k_candidates : int, optional
        Candidate faces kept by approximate methods. By default 64.

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
