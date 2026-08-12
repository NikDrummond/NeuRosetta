"""Functions for computing distances from neuropil surfaces."""

from numpy import ndarray

from ...core import _Mesh
from ...utils.vedo_utils import mesh_surface_depth, surface_distance


def distance_from_neuropil_surface(mesh: _Mesh, points: ndarray) -> ndarray:
    """Compute distances from points to the closest vertices on a mesh.

    Parameters
    ----------
    mesh : _Mesh
        Neuropil mesh object with a .mesh attribute (vedo.Mesh).
    points : ndarray
        Query points with shape (N, 3).

    Returns
    -------
    ndarray
        Array of distances with shape (N,).
    """
    dists, _ = surface_distance(points=points, mesh=mesh.mesh)
    return dists


def neuropil_point_depth(
    mesh: _Mesh, points: ndarray, t: float, surface: str = "inner", norm: bool = True
) -> ndarray:
    """Compute normalized depth of points relative to mesh surfaces.

    Parameters
    ----------
    mesh : _Mesh
        Neuropil mesh object with a .mesh attribute (vedo.Mesh).
    points : ndarray
        Query points with shape (N, 3).
    t : float
        Dot product threshold for classifying inner/outer faces.
    surface : str, optional
        Which surface to compute depth for: "inner" or "outer". By default "inner".
    norm : bool, optional
        If True, normalize distances by sum of inner and outer distances.
        By default True.

    Returns
    -------
    ndarray
        Array of depth values with shape (N,). Values in [0, 1] if normalized.
    """
    return mesh_surface_depth(mesh=mesh.mesh, points=points, t=t, surface=surface, norm=norm)
